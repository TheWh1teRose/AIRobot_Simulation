using System.Collections;
using System.Collections.Generic;
using UnityEngine;



public class Random : MonoBehaviour {
	//private Vector3[] startpoints = new Vector3[] {new Vector3(0.286, 1.171, 2.721), new Vector3(0.222, 1.171, 2.135), new Vector3(-0.251, 1.171, 2.288), new Vector3(-0.324, 1.171, 2.831)};
	private float x;
	private float y;
	private float z;
	// Use this for initialization
	void Start () {
		x = UnityEngine.Random.Range (-0.451f, 0.715f);
		y = 1.121f;
		z = UnityEngine.Random.Range (2.831f, 2.183f);
		Vector3 pos = new Vector3 (x, y, z);
		gameObject.transform.position = pos;
	}

	// Update is called once per frame
	void Update () {

	}
}
