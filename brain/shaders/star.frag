#version 330 core

in float v_brightness;

out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0, 1.0, 1.0, v_brightness * 0.4);
}
