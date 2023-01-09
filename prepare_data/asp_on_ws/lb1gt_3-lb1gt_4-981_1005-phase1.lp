
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(lb1gt_3,130,981). coords(lb1gt_3,251,86,981).  visible(lb1gt_3,981). happensAt(walking(lb1gt_3),981). orientation(lb1gt_4,90,981). coords(lb1gt_4,310,112,981).  visible(lb1gt_4,981). happensAt(inactive(lb1gt_4),981).
orientation(lb1gt_3,132,982). coords(lb1gt_3,248,86,982).  visible(lb1gt_3,982). happensAt(walking(lb1gt_3),982). orientation(lb1gt_4,90,982). coords(lb1gt_4,310,112,982).  visible(lb1gt_4,982). happensAt(inactive(lb1gt_4),982).
orientation(lb1gt_3,134,983). coords(lb1gt_3,244,86,983).  visible(lb1gt_3,983). happensAt(walking(lb1gt_3),983). orientation(lb1gt_4,90,983). coords(lb1gt_4,310,112,983).  visible(lb1gt_4,983). happensAt(inactive(lb1gt_4),983).
orientation(lb1gt_3,134,984). coords(lb1gt_3,240,86,984).  visible(lb1gt_3,984). happensAt(walking(lb1gt_3),984). orientation(lb1gt_4,90,984). coords(lb1gt_4,310,112,984).  visible(lb1gt_4,984). happensAt(inactive(lb1gt_4),984).
orientation(lb1gt_3,134,985). coords(lb1gt_3,236,85,985).  visible(lb1gt_3,985). happensAt(walking(lb1gt_3),985). orientation(lb1gt_4,90,985). coords(lb1gt_4,310,112,985).  visible(lb1gt_4,985). happensAt(inactive(lb1gt_4),985).
orientation(lb1gt_3,134,986). coords(lb1gt_3,233,85,986).  visible(lb1gt_3,986). happensAt(walking(lb1gt_3),986). orientation(lb1gt_4,90,986). coords(lb1gt_4,310,112,986).  visible(lb1gt_4,986). happensAt(inactive(lb1gt_4),986).
orientation(lb1gt_3,134,987). coords(lb1gt_3,231,85,987).  visible(lb1gt_3,987). happensAt(walking(lb1gt_3),987). orientation(lb1gt_4,90,987). coords(lb1gt_4,310,112,987).  visible(lb1gt_4,987). happensAt(inactive(lb1gt_4),987).
orientation(lb1gt_3,134,988). coords(lb1gt_3,230,85,988).  visible(lb1gt_3,988). happensAt(walking(lb1gt_3),988). orientation(lb1gt_4,90,988). coords(lb1gt_4,310,112,988).  visible(lb1gt_4,988). happensAt(inactive(lb1gt_4),988).
orientation(lb1gt_3,134,989). coords(lb1gt_3,229,85,989).  visible(lb1gt_3,989). happensAt(walking(lb1gt_3),989). orientation(lb1gt_4,90,989). coords(lb1gt_4,310,112,989).  visible(lb1gt_4,989). happensAt(inactive(lb1gt_4),989).
orientation(lb1gt_3,134,990). coords(lb1gt_3,227,85,990).  visible(lb1gt_3,990). happensAt(walking(lb1gt_3),990). orientation(lb1gt_4,90,990). coords(lb1gt_4,310,112,990).  visible(lb1gt_4,990). happensAt(inactive(lb1gt_4),990).
orientation(lb1gt_3,134,991). coords(lb1gt_3,226,85,991).  visible(lb1gt_3,991). happensAt(walking(lb1gt_3),991). orientation(lb1gt_4,90,991). coords(lb1gt_4,310,112,991).  visible(lb1gt_4,991). happensAt(inactive(lb1gt_4),991).
orientation(lb1gt_3,134,992). coords(lb1gt_3,225,85,992).  visible(lb1gt_3,992). happensAt(walking(lb1gt_3),992). orientation(lb1gt_4,90,992). coords(lb1gt_4,310,112,992).  visible(lb1gt_4,992). happensAt(inactive(lb1gt_4),992).
orientation(lb1gt_3,134,993). coords(lb1gt_3,223,85,993).  visible(lb1gt_3,993). happensAt(walking(lb1gt_3),993). orientation(lb1gt_4,90,993). coords(lb1gt_4,310,112,993).  visible(lb1gt_4,993). happensAt(inactive(lb1gt_4),993).
orientation(lb1gt_3,134,994). coords(lb1gt_3,221,85,994).  visible(lb1gt_3,994). happensAt(walking(lb1gt_3),994). orientation(lb1gt_4,90,994). coords(lb1gt_4,310,112,994).  visible(lb1gt_4,994). happensAt(inactive(lb1gt_4),994).
orientation(lb1gt_3,134,995). coords(lb1gt_3,217,85,995).  visible(lb1gt_3,995). happensAt(walking(lb1gt_3),995). orientation(lb1gt_4,90,995). coords(lb1gt_4,310,112,995).  visible(lb1gt_4,995). happensAt(inactive(lb1gt_4),995).
orientation(lb1gt_3,134,996). coords(lb1gt_3,214,84,996).  visible(lb1gt_3,996). happensAt(walking(lb1gt_3),996). orientation(lb1gt_4,90,996). coords(lb1gt_4,310,112,996).  visible(lb1gt_4,996). happensAt(inactive(lb1gt_4),996).
orientation(lb1gt_3,134,997). coords(lb1gt_3,211,83,997).  visible(lb1gt_3,997). happensAt(walking(lb1gt_3),997). orientation(lb1gt_4,90,997). coords(lb1gt_4,310,112,997).  visible(lb1gt_4,997). happensAt(inactive(lb1gt_4),997).
orientation(lb1gt_3,134,998). coords(lb1gt_3,208,83,998).  visible(lb1gt_3,998). happensAt(walking(lb1gt_3),998). orientation(lb1gt_4,90,998). coords(lb1gt_4,310,112,998).  visible(lb1gt_4,998). happensAt(inactive(lb1gt_4),998).
orientation(lb1gt_3,134,999). coords(lb1gt_3,205,82,999).  visible(lb1gt_3,999). happensAt(walking(lb1gt_3),999). orientation(lb1gt_4,90,999). coords(lb1gt_4,310,112,999).  visible(lb1gt_4,999). happensAt(inactive(lb1gt_4),999).
orientation(lb1gt_3,134,1000). coords(lb1gt_3,204,82,1000).  visible(lb1gt_3,1000). happensAt(walking(lb1gt_3),1000). orientation(lb1gt_4,90,1000). coords(lb1gt_4,310,112,1000).  visible(lb1gt_4,1000). happensAt(inactive(lb1gt_4),1000).
orientation(lb1gt_3,134,1001). coords(lb1gt_3,203,82,1001).  visible(lb1gt_3,1001). happensAt(walking(lb1gt_3),1001). orientation(lb1gt_4,90,1001). coords(lb1gt_4,310,112,1001).  visible(lb1gt_4,1001). happensAt(inactive(lb1gt_4),1001).
orientation(lb1gt_3,134,1002). coords(lb1gt_3,202,81,1002).  visible(lb1gt_3,1002). happensAt(walking(lb1gt_3),1002). orientation(lb1gt_4,90,1002). coords(lb1gt_4,310,112,1002).  visible(lb1gt_4,1002). happensAt(inactive(lb1gt_4),1002).
orientation(lb1gt_3,134,1003). coords(lb1gt_3,201,80,1003).  visible(lb1gt_3,1003). happensAt(walking(lb1gt_3),1003). orientation(lb1gt_4,90,1003). coords(lb1gt_4,310,112,1003).  visible(lb1gt_4,1003). happensAt(inactive(lb1gt_4),1003).
orientation(lb1gt_3,134,1004). coords(lb1gt_3,201,80,1004).  visible(lb1gt_3,1004). happensAt(walking(lb1gt_3),1004). orientation(lb1gt_4,90,1004). coords(lb1gt_4,310,112,1004).  visible(lb1gt_4,1004). happensAt(inactive(lb1gt_4),1004).
orientation(lb1gt_3,134,1005). coords(lb1gt_3,199,80,1005).  visible(lb1gt_3,1005). happensAt(walking(lb1gt_3),1005). orientation(lb1gt_4,90,1005). coords(lb1gt_4,310,112,1005).  visible(lb1gt_4,1005). happensAt(inactive(lb1gt_4),1005).
#show holdsAt/2.