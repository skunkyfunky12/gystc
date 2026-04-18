#version 330 core

in vec4 v_color;

out vec4 FragColor;

void main()
{
    // Slight boost to edge visibility
    vec3 col = v_color.rgb * 1.3;
    FragColor = vec4(col, v_color.a * 1.5);
}
