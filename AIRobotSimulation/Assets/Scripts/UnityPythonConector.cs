using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using System.Threading;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System;

public class UnityPythonConector : MonoBehaviour {

    public GameObject traget;
    private UdpClient client = null;
    float[,,] positionMatrix;
    int isRestarted = 0;

    float x = 30;
    float y = 30;
    float h = 30;
    int smothing = 8;
    //                         x,   y,    h
    float[] startPositions = {-1.3f,0.5f,1.3f};
    float[] endPositions   = {1.1f,2.9f,3.7f};

    // Use this for initialization
    void Start () {
        client = new UdpClient(5002);

        positionMatrix = new float[Convert.ToInt32(x), Convert.ToInt32(y), Convert.ToInt32(h)];

    }
	
	// Update is called once per frame
	void Update () {
        if(Input.GetKey("f")){
            SceneManager.LoadScene(SceneManager.GetActiveScene().name);
            Time.timeScale = 1f;
            isRestarted = 1;
        }

        UDPSendControls();
    }

    private void UDPSendControls()
    {
        string controls = Input.GetAxis("Horizontal") + ":"
            + -Input.GetAxis("Vertical") + ":"
            + -Input.GetAxis("Height") + ":"  + getGrab();

        IPEndPoint endpoint = new IPEndPoint(IPAddress.Parse("127.0.0.1"), 5002);
        byte[] sendBytes = Encoding.ASCII.GetBytes(controls + "$" + isRestarted + "$" + getPositionInMatrix());
        isRestarted = 0;
        Debug.Log(getPositionInMatrix());
        client.Send(sendBytes, sendBytes.Length, endpoint);
    }

    private String getPositionInMatrix()
    {
        //get the postions from the target. NOTE: some dimentions are switcht for realism
        float targetX = traget.transform.position.x;
        float targetY = traget.transform.position.z;
        float targetH = traget.transform.position.y;

        float distanceToStartX = targetX - startPositions[0];
        float distanceToStartY = targetY - startPositions[1];
        float distanceToStartH = targetH - startPositions[2];

        int positionInPosiotinMatrixX = Convert.ToInt32(Math.Round(distanceToStartX / ((endPositions[0] - startPositions[0]) / x)));
        int positionInPosiotinMatrixY = Convert.ToInt32(Math.Round(distanceToStartY / ((endPositions[1] - startPositions[1]) / y)));
        int positionInPosiotinMatrixH = Convert.ToInt32(Math.Round(distanceToStartH / ((endPositions[2] - startPositions[2]) / h)));

        //Debug.Log(positionInPosiotinMatrixX + ":" + positionInPosiotinMatrixY + ":" + positionInPosiotinMatrixH);

        return (positionInPosiotinMatrixX + ":" + positionInPosiotinMatrixY + ":" + positionInPosiotinMatrixH);

    }

    private int getGrab()
    {
        if (Input.GetKey("g"))
        {
            return 1;
        }else if (Input.GetKey("b"))
        {
            return -1;
        }
        return 0;
    }
}
