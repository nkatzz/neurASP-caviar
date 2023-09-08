
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
orientation(wk2gt_1,133,1005). coords(wk2gt_1,135,41,1005).  visible(wk2gt_1,1005). happensAt(walking(wk2gt_1),1005). orientation(wk2gt_5,150,1005). coords(wk2gt_5,113,116,1005).  visible(wk2gt_5,1005). happensAt(walking(wk2gt_5),1005).
orientation(wk2gt_1,133,1006). coords(wk2gt_1,135,40,1006).  visible(wk2gt_1,1006). happensAt(walking(wk2gt_1),1006). orientation(wk2gt_5,150,1006). coords(wk2gt_5,113,116,1006).  visible(wk2gt_5,1006). happensAt(walking(wk2gt_5),1006).
orientation(wk2gt_1,133,1007). coords(wk2gt_1,135,40,1007).  visible(wk2gt_1,1007). happensAt(walking(wk2gt_1),1007). orientation(wk2gt_5,150,1007). coords(wk2gt_5,113,116,1007).  visible(wk2gt_5,1007). happensAt(walking(wk2gt_5),1007).
orientation(wk2gt_1,133,1008). coords(wk2gt_1,135,39,1008).  visible(wk2gt_1,1008). happensAt(walking(wk2gt_1),1008). orientation(wk2gt_5,150,1008). coords(wk2gt_5,113,118,1008).  visible(wk2gt_5,1008). happensAt(walking(wk2gt_5),1008).
orientation(wk2gt_1,133,1009). coords(wk2gt_1,135,39,1009).  visible(wk2gt_1,1009). happensAt(walking(wk2gt_1),1009). orientation(wk2gt_5,150,1009). coords(wk2gt_5,113,118,1009).  visible(wk2gt_5,1009). happensAt(walking(wk2gt_5),1009).
orientation(wk2gt_1,133,1010). coords(wk2gt_1,135,39,1010).  visible(wk2gt_1,1010). happensAt(walking(wk2gt_1),1010). orientation(wk2gt_5,150,1010). coords(wk2gt_5,114,120,1010).  visible(wk2gt_5,1010). happensAt(walking(wk2gt_5),1010).
orientation(wk2gt_1,133,1011). coords(wk2gt_1,135,39,1011).  visible(wk2gt_1,1011). happensAt(walking(wk2gt_1),1011). orientation(wk2gt_5,150,1011). coords(wk2gt_5,114,121,1011).  visible(wk2gt_5,1011). happensAt(walking(wk2gt_5),1011).
orientation(wk2gt_1,133,1012). coords(wk2gt_1,135,39,1012).  visible(wk2gt_1,1012). happensAt(walking(wk2gt_1),1012). orientation(wk2gt_5,150,1012). coords(wk2gt_5,116,122,1012).  visible(wk2gt_5,1012). happensAt(walking(wk2gt_5),1012).
orientation(wk2gt_1,133,1013). coords(wk2gt_1,135,39,1013).  visible(wk2gt_1,1013). happensAt(walking(wk2gt_1),1013). orientation(wk2gt_5,150,1013). coords(wk2gt_5,118,122,1013).  visible(wk2gt_5,1013). happensAt(walking(wk2gt_5),1013).
orientation(wk2gt_1,133,1014). coords(wk2gt_1,135,39,1014).  visible(wk2gt_1,1014). happensAt(walking(wk2gt_1),1014). orientation(wk2gt_5,150,1014). coords(wk2gt_5,119,123,1014).  visible(wk2gt_5,1014). happensAt(walking(wk2gt_5),1014).
orientation(wk2gt_1,133,1015). coords(wk2gt_1,135,39,1015).  visible(wk2gt_1,1015). happensAt(walking(wk2gt_1),1015). orientation(wk2gt_5,150,1015). coords(wk2gt_5,120,123,1015).  visible(wk2gt_5,1015). happensAt(walking(wk2gt_5),1015).
orientation(wk2gt_1,133,1016). coords(wk2gt_1,136,39,1016).  visible(wk2gt_1,1016). happensAt(walking(wk2gt_1),1016). orientation(wk2gt_5,150,1016). coords(wk2gt_5,122,124,1016).  visible(wk2gt_5,1016). happensAt(walking(wk2gt_5),1016).
#show holdsAt/2.