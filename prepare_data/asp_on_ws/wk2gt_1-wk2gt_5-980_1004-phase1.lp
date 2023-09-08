
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

	orientation(wk2gt_1,133,980). coords(wk2gt_1,130,43,980).  visible(wk2gt_1,980). happensAt(walking(wk2gt_1),980). orientation(wk2gt_5,150,980). coords(wk2gt_5,94,108,980).  visible(wk2gt_5,980). happensAt(walking(wk2gt_5),980).
orientation(wk2gt_1,133,981). coords(wk2gt_1,130,43,981).  visible(wk2gt_1,981). happensAt(walking(wk2gt_1),981). orientation(wk2gt_5,150,981). coords(wk2gt_5,94,108,981).  visible(wk2gt_5,981). happensAt(walking(wk2gt_5),981).
orientation(wk2gt_1,133,982). coords(wk2gt_1,130,43,982).  visible(wk2gt_1,982). happensAt(walking(wk2gt_1),982). orientation(wk2gt_5,150,982). coords(wk2gt_5,95,108,982).  visible(wk2gt_5,982). happensAt(walking(wk2gt_5),982).
orientation(wk2gt_1,133,983). coords(wk2gt_1,130,43,983).  visible(wk2gt_1,983). happensAt(walking(wk2gt_1),983). orientation(wk2gt_5,150,983). coords(wk2gt_5,95,108,983).  visible(wk2gt_5,983). happensAt(walking(wk2gt_5),983).
orientation(wk2gt_1,133,984). coords(wk2gt_1,130,43,984).  visible(wk2gt_1,984). happensAt(walking(wk2gt_1),984). orientation(wk2gt_5,150,984). coords(wk2gt_5,95,109,984).  visible(wk2gt_5,984). happensAt(walking(wk2gt_5),984).
orientation(wk2gt_1,133,985). coords(wk2gt_1,130,43,985).  visible(wk2gt_1,985). happensAt(walking(wk2gt_1),985). orientation(wk2gt_5,150,985). coords(wk2gt_5,95,110,985).  visible(wk2gt_5,985). happensAt(walking(wk2gt_5),985).
orientation(wk2gt_1,133,986). coords(wk2gt_1,130,43,986).  visible(wk2gt_1,986). happensAt(walking(wk2gt_1),986). orientation(wk2gt_5,150,986). coords(wk2gt_5,95,110,986).  visible(wk2gt_5,986). happensAt(walking(wk2gt_5),986).
orientation(wk2gt_1,133,987). coords(wk2gt_1,130,42,987).  visible(wk2gt_1,987). happensAt(walking(wk2gt_1),987). orientation(wk2gt_5,150,987). coords(wk2gt_5,95,112,987).  visible(wk2gt_5,987). happensAt(walking(wk2gt_5),987).
orientation(wk2gt_1,133,988). coords(wk2gt_1,130,43,988).  visible(wk2gt_1,988). happensAt(walking(wk2gt_1),988). orientation(wk2gt_5,150,988). coords(wk2gt_5,97,113,988).  visible(wk2gt_5,988). happensAt(walking(wk2gt_5),988).
orientation(wk2gt_1,133,989). coords(wk2gt_1,131,42,989).  visible(wk2gt_1,989). happensAt(walking(wk2gt_1),989). orientation(wk2gt_5,150,989). coords(wk2gt_5,98,113,989).  visible(wk2gt_5,989). happensAt(walking(wk2gt_5),989).
orientation(wk2gt_1,133,990). coords(wk2gt_1,131,42,990).  visible(wk2gt_1,990). happensAt(walking(wk2gt_1),990). orientation(wk2gt_5,150,990). coords(wk2gt_5,99,114,990).  visible(wk2gt_5,990). happensAt(walking(wk2gt_5),990).
orientation(wk2gt_1,133,991). coords(wk2gt_1,132,42,991).  visible(wk2gt_1,991). happensAt(walking(wk2gt_1),991). orientation(wk2gt_5,150,991). coords(wk2gt_5,101,114,991).  visible(wk2gt_5,991). happensAt(walking(wk2gt_5),991).
orientation(wk2gt_1,133,992). coords(wk2gt_1,132,42,992).  visible(wk2gt_1,992). happensAt(walking(wk2gt_1),992). orientation(wk2gt_5,150,992). coords(wk2gt_5,101,114,992).  visible(wk2gt_5,992). happensAt(walking(wk2gt_5),992).
orientation(wk2gt_1,133,993). coords(wk2gt_1,132,41,993).  visible(wk2gt_1,993). happensAt(walking(wk2gt_1),993). orientation(wk2gt_5,150,993). coords(wk2gt_5,102,115,993).  visible(wk2gt_5,993). happensAt(walking(wk2gt_5),993).
orientation(wk2gt_1,133,994). coords(wk2gt_1,133,41,994).  visible(wk2gt_1,994). happensAt(walking(wk2gt_1),994). orientation(wk2gt_5,150,994). coords(wk2gt_5,102,115,994).  visible(wk2gt_5,994). happensAt(walking(wk2gt_5),994).
orientation(wk2gt_1,133,995). coords(wk2gt_1,133,41,995).  visible(wk2gt_1,995). happensAt(walking(wk2gt_1),995). orientation(wk2gt_5,150,995). coords(wk2gt_5,102,115,995).  visible(wk2gt_5,995). happensAt(walking(wk2gt_5),995).
orientation(wk2gt_1,133,996). coords(wk2gt_1,133,41,996).  visible(wk2gt_1,996). happensAt(walking(wk2gt_1),996). orientation(wk2gt_5,150,996). coords(wk2gt_5,102,115,996).  visible(wk2gt_5,996). happensAt(walking(wk2gt_5),996).
orientation(wk2gt_1,133,997). coords(wk2gt_1,133,41,997).  visible(wk2gt_1,997). happensAt(walking(wk2gt_1),997). orientation(wk2gt_5,150,997). coords(wk2gt_5,102,115,997).  visible(wk2gt_5,997). happensAt(walking(wk2gt_5),997).
orientation(wk2gt_1,133,998). coords(wk2gt_1,134,41,998).  visible(wk2gt_1,998). happensAt(walking(wk2gt_1),998). orientation(wk2gt_5,150,998). coords(wk2gt_5,104,115,998).  visible(wk2gt_5,998). happensAt(walking(wk2gt_5),998).
orientation(wk2gt_1,133,999). coords(wk2gt_1,134,41,999).  visible(wk2gt_1,999). happensAt(walking(wk2gt_1),999). orientation(wk2gt_5,150,999). coords(wk2gt_5,105,116,999).  visible(wk2gt_5,999). happensAt(walking(wk2gt_5),999).
orientation(wk2gt_1,133,1000). coords(wk2gt_1,134,41,1000).  visible(wk2gt_1,1000). happensAt(walking(wk2gt_1),1000). orientation(wk2gt_5,150,1000). coords(wk2gt_5,107,116,1000).  visible(wk2gt_5,1000). happensAt(walking(wk2gt_5),1000).
orientation(wk2gt_1,133,1001). coords(wk2gt_1,134,41,1001).  visible(wk2gt_1,1001). happensAt(walking(wk2gt_1),1001). orientation(wk2gt_5,150,1001). coords(wk2gt_5,108,116,1001).  visible(wk2gt_5,1001). happensAt(walking(wk2gt_5),1001).
orientation(wk2gt_1,133,1002). coords(wk2gt_1,134,41,1002).  visible(wk2gt_1,1002). happensAt(walking(wk2gt_1),1002). orientation(wk2gt_5,150,1002). coords(wk2gt_5,109,116,1002).  visible(wk2gt_5,1002). happensAt(walking(wk2gt_5),1002).
orientation(wk2gt_1,133,1003). coords(wk2gt_1,134,41,1003).  visible(wk2gt_1,1003). happensAt(walking(wk2gt_1),1003). orientation(wk2gt_5,150,1003). coords(wk2gt_5,111,116,1003).  visible(wk2gt_5,1003). happensAt(walking(wk2gt_5),1003).
orientation(wk2gt_1,133,1004). coords(wk2gt_1,135,41,1004).  visible(wk2gt_1,1004). happensAt(walking(wk2gt_1),1004). orientation(wk2gt_5,150,1004). coords(wk2gt_5,112,116,1004).  visible(wk2gt_5,1004). happensAt(walking(wk2gt_5),1004).
#show holdsAt/2.