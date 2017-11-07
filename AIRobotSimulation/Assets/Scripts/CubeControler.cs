using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CubeControler : MonoBehaviour
{

    public GameObject stick;
    private bool isSticking = false;

    // Use this for initialization
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        if (isSticking)
        {
            transform.position = stick.transform.position;
            transform.rotation = stick.transform.rotation;

            if (Input.GetKey("b") && isSticking)
            {
                isSticking = false;
                //Used to avoid a shot effect when releasing an object
                GetComponent<Rigidbody>().useGravity = true;
            }
        }
    }

    void OnTriggerStay(Collider other)
    {
        if (other.gameObject == stick)
        {
            if (Input.GetKey("g") && !isSticking)
            {
                isSticking = true;
                //Used to avoid a shot effect when releasing an object
                GetComponent<Rigidbody>().useGravity = false;
            }
        }
    }
}