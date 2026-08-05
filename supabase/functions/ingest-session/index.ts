// supabase/functions/ingest-session/index.ts
// Supabase Edge Function (Deno/Vercel compatible) for password-gated Web Session Ingestion.
// Acts as the sole enforcement firewall between browser WebAssembly clients and PostgreSQL database.
// The browser NEVER holds database connection strings or credentials directly.

import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.21.0";

// Helper: Verify PBKDF2-HMAC-SHA256 Hash using Web Crypto API
async function verifyPassword(password: string, storedHash: string): Promise<boolean> {
  if (!storedHash || !storedHash.startsWith("pbkdf2:sha256:")) return false;
  const parts = storedHash.split("$");
  if (parts.length !== 3) return false;
  const saltHex = parts[1];
  const expectedHashHex = parts[2];

  const enc = new TextEncoder();
  const saltBytes = new Uint8Array(saltHex.match(/.{1,2}/g)!.map(byte => parseInt(byte, 16)));
  const passwordKey = await crypto.subtle.importKey(
    "raw",
    enc.encode(password),
    { name: "PBKDF2" },
    false,
    ["deriveBits"]
  );

  const derivedBits = await crypto.subtle.deriveBits(
    {
      name: "PBKDF2",
      salt: saltBytes,
      iterations: 100000,
      hash: "SHA-256"
    },
    passwordKey,
    256
  );

  const computedHex = Array.from(new Uint8Array(derivedBits))
    .map(b => b.toString(16).padStart(2, "0"))
    .join("");

  return computedHex === expectedHashHex;
}

// Helper: Hash a new password
async function hashPassword(password: string): Promise<string> {
  const enc = new TextEncoder();
  const saltBytes = crypto.getRandomValues(new Uint8Array(16));
  const saltHex = Array.from(saltBytes).map(b => b.toString(16).padStart(2, "0")).join("");

  const passwordKey = await crypto.subtle.importKey(
    "raw",
    enc.encode(password),
    { name: "PBKDF2" },
    false,
    ["deriveBits"]
  );

  const derivedBits = await crypto.subtle.deriveBits(
    {
      name: "PBKDF2",
      salt: saltBytes,
      iterations: 100000,
      hash: "SHA-256"
    },
    passwordKey,
    256
  );

  const computedHex = Array.from(new Uint8Array(derivedBits))
    .map(b => b.toString(16).padStart(2, "0"))
    .join("");

  return `pbkdf2:sha256:100000$${saltHex}$${computedHex}`;
}

serve(async (req: Request) => {
  // CORS configuration for web browser access
  if (req.method === "OPTIONS") {
    return new Response("ok", {
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
      }
    });
  }

  try {
    const { user_name, password, session_metadata, epoch_features } = await req.json();

    if (!user_name || !password) {
      return new Response(JSON.stringify({ error: "Missing required user_name and password fields." }), { status: 400 });
    }

    // Initialize trusted server-side Supabase client using Service Role Key (NEVER exposed to browsers)
    const supabaseUrl = Deno.env.get("SUPABASE_URL") ?? "";
    const supabaseServiceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
    const supabase = createClient(supabaseUrl, supabaseServiceKey, { auth: { persistSession: false } });

    // Step 1: Query existing user profile server-side
    const { data: userProfile, error: profileErr } = await supabase
      .from("user_profiles")
      .select("user_name, password_hash")
      .eq("user_name", user_name)
      .single();

    let passwordHashToSave = null;
    if (userProfile && userProfile.password_hash) {
      // Step 2: Existing protected subject — verify password BEFORE writing
      const isValid = await verifyPassword(password, userProfile.password_hash);
      if (!isValid) {
        console.warn(`[Web Endpoint] Data integrity rejection: invalid password for subject '${user_name}'.`);
        return new Response(JSON.stringify({ 
          error: "Data Integrity Error: Incorrect password for existing subject account. Ingestion blocked to prevent silent data merging/corruption." 
        }), { status: 403 });
      }
    } else {
      // Step 3: First time user_name is seen (or legacy unhashed user) -> hash password and assign created_via
      passwordHashToSave = await hashPassword(password);
      if (!userProfile) {
        await supabase.from("user_profiles").insert({
          user_name: user_name,
          birth_date: session_metadata?.user_profile?.birth_date || "1900-01-01",
          gender_sex: session_metadata?.user_profile?.gender_sex || "Unknown",
          weight_kg: session_metadata?.user_profile?.weight_kg || null,
          height_cm: session_metadata?.user_profile?.height_cm || null,
          updated_at: Date.now() / 1000,
          password_hash: passwordHashToSave,
          created_via: "web_endpoint"
        });
      } else if (passwordHashToSave) {
        await supabase.from("user_profiles").update({ password_hash: passwordHashToSave }).eq("user_name", user_name);
      }
    }

    // Step 4: Password verified! Proceed with session and epoch insertions
    if (session_metadata) {
      const sessionId = session_metadata.session_id;
      await supabase.from("sessions").upsert({
        session_id: sessionId,
        user_name: user_name,
        start_time: session_metadata.start_time,
        start_time_str: session_metadata.start_time_str,
        end_time: session_metadata.end_time,
        end_time_str: session_metadata.end_time_str,
        mode: session_metadata.mode,
        calibration_rms_left: session_metadata.calibration?.left_rms_reference ?? null,
        calibration_rms_right: session_metadata.calibration?.right_rms_reference ?? null,
        processing_version: session_metadata.processing_version || "phase11_edge_padding"
      });
    }

    if (Array.isArray(epoch_features) && epoch_features.length > 0) {
      const enrichedEpochs = epoch_features.map((item: any) => ({
        ...item,
        session_id: item.session_id || session_metadata?.session_id
      }));
      const { error: insertErr } = await supabase.from("epoch_features").upsert(
        enrichedEpochs, 
        { onConflict: "session_id,epoch_index" }
      );
      if (insertErr) {
        return new Response(JSON.stringify({ error: insertErr.message }), { status: 500 });
      }
    }

    return new Response(JSON.stringify({ 
      status: "success", 
      message: `Successfully authenticated subject '${user_name}' and ingested session telemetry.` 
    }), { 
      status: 200, 
      headers: { "Content-Type": "application/json" } 
    });

  } catch (err) {
    return new Response(JSON.stringify({ error: err.toString() }), { status: 500 });
  }
});
