using UnityEngine;
using UnityEngine.InputSystem;

public class FlyCameraController : MonoBehaviour
{
    public float moveSpeed = 50f;
    public float fastMultiplier = 3f;
    public float lookSensitivity = 0.1f;

    float pitch;
    float yaw;

    void Start()
    {
        Vector3 angles = transform.eulerAngles;
        pitch = angles.x;
        yaw = angles.y;
    }

    void Update()
    {
        HandleLook();
        HandleMovement();
        HandleCursor();
    }

    void HandleLook()
    {
        var mouse = Mouse.current;
        if (mouse == null) return;

        if (!mouse.rightButton.isPressed)
            return;

        Vector2 delta = mouse.delta.ReadValue();

        yaw += delta.x * lookSensitivity;
        pitch -= delta.y * lookSensitivity;
        pitch = Mathf.Clamp(pitch, -90f, 90f);

        transform.eulerAngles = new Vector3(pitch, yaw, 0f);
    }

    void HandleMovement()
    {
        var keyboard = Keyboard.current;
        if (keyboard == null) return;

        float speed = moveSpeed;
        if (keyboard.leftShiftKey.isPressed)
            speed *= fastMultiplier;

        Vector3 move = Vector3.zero;

        // Forward/back
        if (keyboard.wKey.isPressed || keyboard.upArrowKey.isPressed)
            move += transform.forward;
        if (keyboard.sKey.isPressed || keyboard.downArrowKey.isPressed)
            move -= transform.forward;

        // Left/right
        if (keyboard.aKey.isPressed || keyboard.leftArrowKey.isPressed)
            move -= transform.right;
        if (keyboard.dKey.isPressed || keyboard.rightArrowKey.isPressed)
            move += transform.right;

        // Up/down
        if (keyboard.eKey.isPressed)
            move += Vector3.up;
        if (keyboard.qKey.isPressed)
            move -= Vector3.up;

        transform.position += move.normalized * speed * Time.deltaTime;
    }

    void HandleCursor()
    {
        var keyboard = Keyboard.current;
        var mouse = Mouse.current;

        if (keyboard != null && keyboard.escapeKey.wasPressedThisFrame)
        {
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }

        if (mouse != null && mouse.leftButton.wasPressedThisFrame)
        {
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }
    }
}
