#version 330 core

in vec3 v_color;
in vec2 v_uv;

out vec4 FragColor;

void main()
{
    float dist = length(v_uv);
    if (dist > 1.0) discard;

    // Layer 1: white-hot inner core
    float core = smoothstep(0.2, 0.0, dist);

    // Layer 2: saturated color ring
    float ring = smoothstep(0.55, 0.1, dist);

    // Layer 3: wide outer halo (bioluminescence)
    float halo = exp(-dist * 1.2);

    // Combine: white core fading into color, then soft halo
    vec3 col = mix(v_color * 1.4, vec3(1.0), core) * ring + v_color * halo * 0.3;

    // Strong alpha with soft edge
    float alpha = halo * 0.85;

    FragColor = vec4(col, alpha);
}
