using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ControlArm : MonoBehaviour {

    public float moveSpeed = 1;
    public float roationSpeed = 1;
    public GameObject leftGrap;
    public GameObject rightGrap;

    // Use this for initialization
    void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
        transform.Translate(moveSpeed * Input.GetAxis("Horizontal")*Time.deltaTime,
            moveSpeed * -Input.GetAxis("Vertical") * Time.deltaTime,
            moveSpeed * -Input.GetAxis("Height") * Time.deltaTime);
        transform.Rotate(roationSpeed * Input.GetAxis("Rotation") * Time.deltaTime,
            0f,
            0f);
    }
}
