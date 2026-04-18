#version 330 core

in float v_brightness;

out vec4 FragColor;

void main()
{
    // Soft circular point
    vec2 pc = gl_PointCoord * 2.0 - 1.0;
    float dist = length(pc);
    if (dist > 1.0) discard;

    float glow = exp(-dist * 2.5);

    // Slight warm/cool tint based on brightness
    vec3 tint = mix(vec3(0.8, 0.85, 1.0), vec3(1.0, 0.95, 0.8), v_brightness);

    FragColor = vec4(tint, v_brightness * glow * 0.6);
}
