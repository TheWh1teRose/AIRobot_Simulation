using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RoboAgent : Agent {

	public int resWidth = 50; 
	public int resHeight = 50;

	public Camera camera = null;
    public GameObject target = null;
    public GameObject cube = null;
    public GameObject stick = null;

    bool isSticking = false;


    public override List<float> CollectState()
	{
		List<float> state = new List<float>();

		RenderTexture rt = new RenderTexture(resWidth, resHeight, 24);
		camera.targetTexture = rt;
		Texture2D screenShot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24, false);
		camera.Render();
		RenderTexture.active = rt;
		Color[] pix = screenShot.GetPixels(0,0,resWidth,resHeight);
		camera.targetTexture = null;
		RenderTexture.active = null; // JC: added to avoid errors

		foreach (var color in pix) {
			state.Add (color.r);
			state.Add (color.g);
			state.Add (color.b);
		}
	

		return state;
	}

	public override void AgentStep(float[] act)
	{
        Debug.Log(cube.transform.position);
        if (isSticking)
        {
            cube.transform.position = stick.transform.position;
            cube.transform.rotation = stick.transform.rotation;
        }
        switch ((int)act[0])
        {
            case 0:
                break;
            case 1:
                target.transform.Translate(new Vector3(0.01f,0f,0f));
                break;
        }

        switch ((int)act[1])
        {
            case 0:
                break;
            case 1:
                target.transform.Translate(new Vector3(-0.01f, 0f, 0f));
                break;
        }

        switch ((int)act[2])
        {
            case 0:
                break;
            case 1:
                target.transform.Translate(new Vector3(0f, 0.005f, 0f));
                break;
        }

        switch ((int)act[3])
        {
            case 0:
                break;
            case 1:
                target.transform.Translate(new Vector3(0f, -0.01f, 0f));
                break;
        }
        
        switch ((int)act[4])
        {
            case 0:
                break;
            case 1:
                target.transform.Translate(new Vector3(0f, 0f, 0.01f));
                break;
        }

        switch ((int)act[5])
        {
            case 0:
                break;
            case 1:
                target.transform.Translate(new Vector3(0f, 0f, -0.01f));
                break;
        }

        switch ((int)act[6])
        {
            case 0:
                break;
            case 1:
                isSticking = true;
                cube.GetComponent<Rigidbody>().isKinematic = true;
                break;
        }

        switch ((int)act[7])
        {
            case 0:
                break;
            case 1:
                isSticking = false;
                cube.GetComponent<Rigidbody>().isKinematic = false;
                break;
        }

        Debug.Log((cube.transform.position.y));

        if ((cube.transform.position.x <= 3f && cube.transform.position.x >= 2.8f) && (cube.transform.position.z >= -12f && cube.transform.position.z <= -11.8f) && (cube.transform.position.y <= 6.1f))
        {
            Debug.Log("Done");
            
            done = true;
            return;
        }

    }

	public override void AgentReset()
	{
        
        float x = UnityEngine.Random.Range(3.3f, 4.7f);
        float y = 5.9f;
        float z = UnityEngine.Random.Range(-11.5f, -12.2f);
        Vector3 pos = new Vector3(x, y, z);
        cube.transform.position = pos;
    }

	public override void AgentOnDone()
	{
        
	}
}
