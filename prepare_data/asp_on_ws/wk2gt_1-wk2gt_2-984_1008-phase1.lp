
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

	orientation(wk2gt_1,133,984). coords(wk2gt_1,130,43,984).  visible(wk2gt_1,984). happensAt(walking(wk2gt_1),984). orientation(wk2gt_2,140,984). coords(wk2gt_2,97,65,984).  visible(wk2gt_2,984). happensAt(inactive(wk2gt_2),984).
orientation(wk2gt_1,133,985). coords(wk2gt_1,130,43,985).  visible(wk2gt_1,985). happensAt(walking(wk2gt_1),985). orientation(wk2gt_2,140,985). coords(wk2gt_2,97,65,985).  visible(wk2gt_2,985). happensAt(inactive(wk2gt_2),985).
orientation(wk2gt_1,133,986). coords(wk2gt_1,130,43,986).  visible(wk2gt_1,986). happensAt(walking(wk2gt_1),986). orientation(wk2gt_2,140,986). coords(wk2gt_2,97,65,986).  visible(wk2gt_2,986). happensAt(inactive(wk2gt_2),986).
orientation(wk2gt_1,133,987). coords(wk2gt_1,130,42,987).  visible(wk2gt_1,987). happensAt(walking(wk2gt_1),987). orientation(wk2gt_2,140,987). coords(wk2gt_2,97,65,987).  visible(wk2gt_2,987). happensAt(inactive(wk2gt_2),987).
orientation(wk2gt_1,133,988). coords(wk2gt_1,130,43,988).  visible(wk2gt_1,988). happensAt(walking(wk2gt_1),988). orientation(wk2gt_2,140,988). coords(wk2gt_2,97,65,988).  visible(wk2gt_2,988). happensAt(inactive(wk2gt_2),988).
orientation(wk2gt_1,133,989). coords(wk2gt_1,131,42,989).  visible(wk2gt_1,989). happensAt(walking(wk2gt_1),989). orientation(wk2gt_2,140,989). coords(wk2gt_2,97,65,989).  visible(wk2gt_2,989). happensAt(inactive(wk2gt_2),989).
orientation(wk2gt_1,133,990). coords(wk2gt_1,131,42,990).  visible(wk2gt_1,990). happensAt(walking(wk2gt_1),990). orientation(wk2gt_2,140,990). coords(wk2gt_2,97,65,990).  visible(wk2gt_2,990). happensAt(inactive(wk2gt_2),990).
orientation(wk2gt_1,133,991). coords(wk2gt_1,132,42,991).  visible(wk2gt_1,991). happensAt(walking(wk2gt_1),991). orientation(wk2gt_2,140,991). coords(wk2gt_2,97,65,991).  visible(wk2gt_2,991). happensAt(inactive(wk2gt_2),991).
orientation(wk2gt_1,133,992). coords(wk2gt_1,132,42,992).  visible(wk2gt_1,992). happensAt(walking(wk2gt_1),992). orientation(wk2gt_2,140,992). coords(wk2gt_2,97,65,992).  visible(wk2gt_2,992). happensAt(inactive(wk2gt_2),992).
orientation(wk2gt_1,133,993). coords(wk2gt_1,132,41,993).  visible(wk2gt_1,993). happensAt(walking(wk2gt_1),993). orientation(wk2gt_2,140,993). coords(wk2gt_2,97,65,993).  visible(wk2gt_2,993). happensAt(inactive(wk2gt_2),993).
orientation(wk2gt_1,133,994). coords(wk2gt_1,133,41,994).  visible(wk2gt_1,994). happensAt(walking(wk2gt_1),994). orientation(wk2gt_2,140,994). coords(wk2gt_2,97,65,994).  visible(wk2gt_2,994). happensAt(inactive(wk2gt_2),994).
orientation(wk2gt_1,133,995). coords(wk2gt_1,133,41,995).  visible(wk2gt_1,995). happensAt(walking(wk2gt_1),995). orientation(wk2gt_2,140,995). coords(wk2gt_2,97,65,995).  visible(wk2gt_2,995). happensAt(inactive(wk2gt_2),995).
orientation(wk2gt_1,133,996). coords(wk2gt_1,133,41,996).  visible(wk2gt_1,996). happensAt(walking(wk2gt_1),996). orientation(wk2gt_2,140,996). coords(wk2gt_2,97,65,996).  visible(wk2gt_2,996). happensAt(inactive(wk2gt_2),996).
orientation(wk2gt_1,133,997). coords(wk2gt_1,133,41,997).  visible(wk2gt_1,997). happensAt(walking(wk2gt_1),997). orientation(wk2gt_2,140,997). coords(wk2gt_2,97,65,997).  visible(wk2gt_2,997). happensAt(inactive(wk2gt_2),997).
orientation(wk2gt_1,133,998). coords(wk2gt_1,134,41,998).  visible(wk2gt_1,998). happensAt(walking(wk2gt_1),998). orientation(wk2gt_2,140,998). coords(wk2gt_2,97,65,998).  visible(wk2gt_2,998). happensAt(inactive(wk2gt_2),998).
orientation(wk2gt_1,133,999). coords(wk2gt_1,134,41,999).  visible(wk2gt_1,999). happensAt(walking(wk2gt_1),999). orientation(wk2gt_2,140,999). coords(wk2gt_2,97,65,999).  visible(wk2gt_2,999). happensAt(inactive(wk2gt_2),999).
orientation(wk2gt_1,133,1000). coords(wk2gt_1,134,41,1000).  visible(wk2gt_1,1000). happensAt(walking(wk2gt_1),1000). orientation(wk2gt_2,140,1000). coords(wk2gt_2,97,65,1000).  visible(wk2gt_2,1000). happensAt(inactive(wk2gt_2),1000).
orientation(wk2gt_1,133,1001). coords(wk2gt_1,134,41,1001).  visible(wk2gt_1,1001). happensAt(walking(wk2gt_1),1001). orientation(wk2gt_2,140,1001). coords(wk2gt_2,97,65,1001).  visible(wk2gt_2,1001). happensAt(inactive(wk2gt_2),1001).
orientation(wk2gt_1,133,1002). coords(wk2gt_1,134,41,1002).  visible(wk2gt_1,1002). happensAt(walking(wk2gt_1),1002). orientation(wk2gt_2,140,1002). coords(wk2gt_2,97,65,1002).  visible(wk2gt_2,1002). happensAt(inactive(wk2gt_2),1002).
orientation(wk2gt_1,133,1003). coords(wk2gt_1,134,41,1003).  visible(wk2gt_1,1003). happensAt(walking(wk2gt_1),1003). orientation(wk2gt_2,140,1003). coords(wk2gt_2,97,65,1003).  visible(wk2gt_2,1003). happensAt(inactive(wk2gt_2),1003).
orientation(wk2gt_1,133,1004). coords(wk2gt_1,135,41,1004).  visible(wk2gt_1,1004). happensAt(walking(wk2gt_1),1004). orientation(wk2gt_2,140,1004). coords(wk2gt_2,97,65,1004).  visible(wk2gt_2,1004). happensAt(inactive(wk2gt_2),1004).
orientation(wk2gt_1,133,1005). coords(wk2gt_1,135,41,1005).  visible(wk2gt_1,1005). happensAt(walking(wk2gt_1),1005). orientation(wk2gt_2,140,1005). coords(wk2gt_2,97,65,1005).  visible(wk2gt_2,1005). happensAt(inactive(wk2gt_2),1005).
orientation(wk2gt_1,133,1006). coords(wk2gt_1,135,40,1006).  visible(wk2gt_1,1006). happensAt(walking(wk2gt_1),1006). orientation(wk2gt_2,140,1006). coords(wk2gt_2,97,65,1006).  visible(wk2gt_2,1006). happensAt(inactive(wk2gt_2),1006).
orientation(wk2gt_1,133,1007). coords(wk2gt_1,135,40,1007).  visible(wk2gt_1,1007). happensAt(walking(wk2gt_1),1007). orientation(wk2gt_2,140,1007). coords(wk2gt_2,97,65,1007).  visible(wk2gt_2,1007). happensAt(inactive(wk2gt_2),1007).
orientation(wk2gt_1,133,1008). coords(wk2gt_1,135,39,1008).  visible(wk2gt_1,1008). happensAt(walking(wk2gt_1),1008). orientation(wk2gt_2,140,1008). coords(wk2gt_2,97,65,1008).  visible(wk2gt_2,1008). happensAt(inactive(wk2gt_2),1008).
#show holdsAt/2.