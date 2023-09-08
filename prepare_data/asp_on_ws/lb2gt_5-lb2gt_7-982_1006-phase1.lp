
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

	orientation(lb2gt_5,152,982). coords(lb2gt_5,110,133,982).  visible(lb2gt_5,982). happensAt(walking(lb2gt_5),982). orientation(lb2gt_7,0,982). coords(lb2gt_7,100,263,982).  visible(lb2gt_7,982). happensAt(walking(lb2gt_7),982).
orientation(lb2gt_5,152,983). coords(lb2gt_5,111,133,983).  visible(lb2gt_5,983). happensAt(walking(lb2gt_5),983). orientation(lb2gt_7,0,983). coords(lb2gt_7,100,263,983).  visible(lb2gt_7,983). happensAt(walking(lb2gt_7),983).
orientation(lb2gt_5,152,984). coords(lb2gt_5,111,133,984).  visible(lb2gt_5,984). happensAt(walking(lb2gt_5),984). orientation(lb2gt_7,0,984). coords(lb2gt_7,99,263,984).  visible(lb2gt_7,984). happensAt(walking(lb2gt_7),984).
orientation(lb2gt_5,152,985). coords(lb2gt_5,111,133,985).  visible(lb2gt_5,985). happensAt(walking(lb2gt_5),985). orientation(lb2gt_7,0,985). coords(lb2gt_7,99,263,985).  visible(lb2gt_7,985). happensAt(walking(lb2gt_7),985).
orientation(lb2gt_5,152,986). coords(lb2gt_5,112,133,986).  visible(lb2gt_5,986). happensAt(walking(lb2gt_5),986). orientation(lb2gt_7,0,986). coords(lb2gt_7,99,263,986).  visible(lb2gt_7,986). happensAt(walking(lb2gt_7),986).
orientation(lb2gt_5,152,987). coords(lb2gt_5,114,133,987).  visible(lb2gt_5,987). happensAt(walking(lb2gt_5),987). orientation(lb2gt_7,0,987). coords(lb2gt_7,99,263,987).  visible(lb2gt_7,987). happensAt(walking(lb2gt_7),987).
orientation(lb2gt_5,152,988). coords(lb2gt_5,116,135,988).  visible(lb2gt_5,988). happensAt(walking(lb2gt_5),988). orientation(lb2gt_7,0,988). coords(lb2gt_7,99,263,988).  visible(lb2gt_7,988). happensAt(walking(lb2gt_7),988).
orientation(lb2gt_5,152,989). coords(lb2gt_5,118,135,989).  visible(lb2gt_5,989). happensAt(walking(lb2gt_5),989). orientation(lb2gt_7,0,989). coords(lb2gt_7,99,263,989).  visible(lb2gt_7,989). happensAt(walking(lb2gt_7),989).
orientation(lb2gt_5,152,990). coords(lb2gt_5,119,135,990).  visible(lb2gt_5,990). happensAt(walking(lb2gt_5),990). orientation(lb2gt_7,0,990). coords(lb2gt_7,99,263,990).  visible(lb2gt_7,990). happensAt(walking(lb2gt_7),990).
orientation(lb2gt_5,152,991). coords(lb2gt_5,120,135,991).  visible(lb2gt_5,991). happensAt(walking(lb2gt_5),991). orientation(lb2gt_7,0,991). coords(lb2gt_7,99,263,991).  visible(lb2gt_7,991). happensAt(walking(lb2gt_7),991).
orientation(lb2gt_5,152,992). coords(lb2gt_5,122,136,992).  visible(lb2gt_5,992). happensAt(walking(lb2gt_5),992). orientation(lb2gt_7,0,992). coords(lb2gt_7,99,263,992).  visible(lb2gt_7,992). happensAt(walking(lb2gt_7),992).
orientation(lb2gt_5,152,993). coords(lb2gt_5,123,136,993).  visible(lb2gt_5,993). happensAt(walking(lb2gt_5),993). orientation(lb2gt_7,0,993). coords(lb2gt_7,99,263,993).  visible(lb2gt_7,993). happensAt(walking(lb2gt_7),993).
orientation(lb2gt_5,152,994). coords(lb2gt_5,124,137,994).  visible(lb2gt_5,994). happensAt(walking(lb2gt_5),994). orientation(lb2gt_7,0,994). coords(lb2gt_7,99,263,994).  visible(lb2gt_7,994). happensAt(walking(lb2gt_7),994).
orientation(lb2gt_5,152,995). coords(lb2gt_5,124,137,995).  visible(lb2gt_5,995). happensAt(walking(lb2gt_5),995). orientation(lb2gt_7,0,995). coords(lb2gt_7,99,264,995).  visible(lb2gt_7,995). happensAt(walking(lb2gt_7),995).
orientation(lb2gt_5,152,996). coords(lb2gt_5,124,137,996).  visible(lb2gt_5,996). happensAt(walking(lb2gt_5),996). orientation(lb2gt_7,0,996). coords(lb2gt_7,99,264,996).  visible(lb2gt_7,996). happensAt(walking(lb2gt_7),996).
orientation(lb2gt_5,152,997). coords(lb2gt_5,124,137,997).  visible(lb2gt_5,997). happensAt(walking(lb2gt_5),997). orientation(lb2gt_7,0,997). coords(lb2gt_7,99,264,997).  visible(lb2gt_7,997). happensAt(walking(lb2gt_7),997).
orientation(lb2gt_5,152,998). coords(lb2gt_5,125,137,998).  visible(lb2gt_5,998). happensAt(walking(lb2gt_5),998). orientation(lb2gt_7,0,998). coords(lb2gt_7,99,264,998).  visible(lb2gt_7,998). happensAt(walking(lb2gt_7),998).
orientation(lb2gt_5,152,999). coords(lb2gt_5,125,138,999).  visible(lb2gt_5,999). happensAt(walking(lb2gt_5),999). orientation(lb2gt_7,0,999). coords(lb2gt_7,98,264,999).  visible(lb2gt_7,999). happensAt(walking(lb2gt_7),999).
orientation(lb2gt_5,152,1000). coords(lb2gt_5,125,139,1000).  visible(lb2gt_5,1000). happensAt(walking(lb2gt_5),1000). orientation(lb2gt_7,0,1000). coords(lb2gt_7,98,264,1000).  visible(lb2gt_7,1000). happensAt(walking(lb2gt_7),1000).
orientation(lb2gt_5,152,1001). coords(lb2gt_5,128,140,1001).  visible(lb2gt_5,1001). happensAt(walking(lb2gt_5),1001). orientation(lb2gt_7,0,1001). coords(lb2gt_7,98,264,1001).  visible(lb2gt_7,1001). happensAt(walking(lb2gt_7),1001).
orientation(lb2gt_5,152,1002). coords(lb2gt_5,130,141,1002).  visible(lb2gt_5,1002). happensAt(walking(lb2gt_5),1002). orientation(lb2gt_7,0,1002). coords(lb2gt_7,98,264,1002).  visible(lb2gt_7,1002). happensAt(walking(lb2gt_7),1002).
orientation(lb2gt_5,152,1003). coords(lb2gt_5,131,141,1003).  visible(lb2gt_5,1003). happensAt(walking(lb2gt_5),1003). orientation(lb2gt_7,0,1003). coords(lb2gt_7,98,263,1003).  visible(lb2gt_7,1003). happensAt(walking(lb2gt_7),1003).
orientation(lb2gt_5,152,1004). coords(lb2gt_5,133,142,1004).  visible(lb2gt_5,1004). happensAt(walking(lb2gt_5),1004). orientation(lb2gt_7,0,1004). coords(lb2gt_7,98,263,1004).  visible(lb2gt_7,1004). happensAt(walking(lb2gt_7),1004).
orientation(lb2gt_5,152,1005). coords(lb2gt_5,135,143,1005).  visible(lb2gt_5,1005). happensAt(walking(lb2gt_5),1005). orientation(lb2gt_7,0,1005). coords(lb2gt_7,98,263,1005).  visible(lb2gt_7,1005). happensAt(walking(lb2gt_7),1005).
orientation(lb2gt_5,152,1006). coords(lb2gt_5,135,143,1006).  visible(lb2gt_5,1006). happensAt(walking(lb2gt_5),1006). orientation(lb2gt_7,0,1006). coords(lb2gt_7,98,263,1006).  visible(lb2gt_7,1006). happensAt(walking(lb2gt_7),1006).
#show holdsAt/2.