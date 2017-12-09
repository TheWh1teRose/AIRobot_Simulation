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
        int horizontal = 0;
        int vertical = 0;
        int height = 0;
        if(Input.GetAxis("Horizontal")>0){horizontal=1;}
        if(Input.GetAxis("Horizontal")<0){horizontal=-1;}
        if(Input.GetAxis("Vertical")>0){vertical=1;}
        if(Input.GetAxis("Vertical")<0){vertical=-1;}
        if(Input.GetAxis("Height")>0){height=1;}
        if(Input.GetAxis("Height")<0){height=-1;}

        transform.Translate(moveSpeed * horizontal*Time.deltaTime,
            moveSpeed * -vertical * Time.deltaTime,
            moveSpeed * -height * Time.deltaTime);
        transform.Rotate(roationSpeed * Input.GetAxis("Rotation") * Time.deltaTime,
            0f,
            0f);
    }
}
