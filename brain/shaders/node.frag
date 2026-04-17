#version 330 core

in vec3 v_color;
in vec2 v_uv;

out vec4 FragColor;

void main()
{
    // Distance from quad center
    float dist = length(v_uv);

    // Discard pixels outside the circle
    if (dist > 1.0) discard;

    // White-hot center core
    float core = smoothstep(0.3, 0.0, dist);

    // Blend color with white at the center
    vec3 col = mix(v_color, vec3(1.0), core);

    // Exponential alpha falloff
    float alpha = exp(-dist * 2.5);

    FragColor = vec4(col, alpha);
}
