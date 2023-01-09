
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

	orientation(wk2gt_2,140,992). coords(wk2gt_2,97,65,992).  visible(wk2gt_2,992). happensAt(inactive(wk2gt_2),992). orientation(wk2gt_5,150,992). coords(wk2gt_5,101,114,992).  visible(wk2gt_5,992). happensAt(walking(wk2gt_5),992).
orientation(wk2gt_2,140,993). coords(wk2gt_2,97,65,993).  visible(wk2gt_2,993). happensAt(inactive(wk2gt_2),993). orientation(wk2gt_5,150,993). coords(wk2gt_5,102,115,993).  visible(wk2gt_5,993). happensAt(walking(wk2gt_5),993).
orientation(wk2gt_2,140,994). coords(wk2gt_2,97,65,994).  visible(wk2gt_2,994). happensAt(inactive(wk2gt_2),994). orientation(wk2gt_5,150,994). coords(wk2gt_5,102,115,994).  visible(wk2gt_5,994). happensAt(walking(wk2gt_5),994).
orientation(wk2gt_2,140,995). coords(wk2gt_2,97,65,995).  visible(wk2gt_2,995). happensAt(inactive(wk2gt_2),995). orientation(wk2gt_5,150,995). coords(wk2gt_5,102,115,995).  visible(wk2gt_5,995). happensAt(walking(wk2gt_5),995).
orientation(wk2gt_2,140,996). coords(wk2gt_2,97,65,996).  visible(wk2gt_2,996). happensAt(inactive(wk2gt_2),996). orientation(wk2gt_5,150,996). coords(wk2gt_5,102,115,996).  visible(wk2gt_5,996). happensAt(walking(wk2gt_5),996).
orientation(wk2gt_2,140,997). coords(wk2gt_2,97,65,997).  visible(wk2gt_2,997). happensAt(inactive(wk2gt_2),997). orientation(wk2gt_5,150,997). coords(wk2gt_5,102,115,997).  visible(wk2gt_5,997). happensAt(walking(wk2gt_5),997).
orientation(wk2gt_2,140,998). coords(wk2gt_2,97,65,998).  visible(wk2gt_2,998). happensAt(inactive(wk2gt_2),998). orientation(wk2gt_5,150,998). coords(wk2gt_5,104,115,998).  visible(wk2gt_5,998). happensAt(walking(wk2gt_5),998).
orientation(wk2gt_2,140,999). coords(wk2gt_2,97,65,999).  visible(wk2gt_2,999). happensAt(inactive(wk2gt_2),999). orientation(wk2gt_5,150,999). coords(wk2gt_5,105,116,999).  visible(wk2gt_5,999). happensAt(walking(wk2gt_5),999).
orientation(wk2gt_2,140,1000). coords(wk2gt_2,97,65,1000).  visible(wk2gt_2,1000). happensAt(inactive(wk2gt_2),1000). orientation(wk2gt_5,150,1000). coords(wk2gt_5,107,116,1000).  visible(wk2gt_5,1000). happensAt(walking(wk2gt_5),1000).
orientation(wk2gt_2,140,1001). coords(wk2gt_2,97,65,1001).  visible(wk2gt_2,1001). happensAt(inactive(wk2gt_2),1001). orientation(wk2gt_5,150,1001). coords(wk2gt_5,108,116,1001).  visible(wk2gt_5,1001). happensAt(walking(wk2gt_5),1001).
orientation(wk2gt_2,140,1002). coords(wk2gt_2,97,65,1002).  visible(wk2gt_2,1002). happensAt(inactive(wk2gt_2),1002). orientation(wk2gt_5,150,1002). coords(wk2gt_5,109,116,1002).  visible(wk2gt_5,1002). happensAt(walking(wk2gt_5),1002).
orientation(wk2gt_2,140,1003). coords(wk2gt_2,97,65,1003).  visible(wk2gt_2,1003). happensAt(inactive(wk2gt_2),1003). orientation(wk2gt_5,150,1003). coords(wk2gt_5,111,116,1003).  visible(wk2gt_5,1003). happensAt(walking(wk2gt_5),1003).
orientation(wk2gt_2,140,1004). coords(wk2gt_2,97,65,1004).  visible(wk2gt_2,1004). happensAt(inactive(wk2gt_2),1004). orientation(wk2gt_5,150,1004). coords(wk2gt_5,112,116,1004).  visible(wk2gt_5,1004). happensAt(walking(wk2gt_5),1004).
orientation(wk2gt_2,140,1005). coords(wk2gt_2,97,65,1005).  visible(wk2gt_2,1005). happensAt(inactive(wk2gt_2),1005). orientation(wk2gt_5,150,1005). coords(wk2gt_5,113,116,1005).  visible(wk2gt_5,1005). happensAt(walking(wk2gt_5),1005).
orientation(wk2gt_2,140,1006). coords(wk2gt_2,97,65,1006).  visible(wk2gt_2,1006). happensAt(inactive(wk2gt_2),1006). orientation(wk2gt_5,150,1006). coords(wk2gt_5,113,116,1006).  visible(wk2gt_5,1006). happensAt(walking(wk2gt_5),1006).
orientation(wk2gt_2,140,1007). coords(wk2gt_2,97,65,1007).  visible(wk2gt_2,1007). happensAt(inactive(wk2gt_2),1007). orientation(wk2gt_5,150,1007). coords(wk2gt_5,113,116,1007).  visible(wk2gt_5,1007). happensAt(walking(wk2gt_5),1007).
orientation(wk2gt_2,140,1008). coords(wk2gt_2,97,65,1008).  visible(wk2gt_2,1008). happensAt(inactive(wk2gt_2),1008). orientation(wk2gt_5,150,1008). coords(wk2gt_5,113,118,1008).  visible(wk2gt_5,1008). happensAt(walking(wk2gt_5),1008).
orientation(wk2gt_2,140,1009). coords(wk2gt_2,97,65,1009).  visible(wk2gt_2,1009). happensAt(inactive(wk2gt_2),1009). orientation(wk2gt_5,150,1009). coords(wk2gt_5,113,118,1009).  visible(wk2gt_5,1009). happensAt(walking(wk2gt_5),1009).
orientation(wk2gt_2,140,1010). coords(wk2gt_2,97,65,1010).  visible(wk2gt_2,1010). happensAt(inactive(wk2gt_2),1010). orientation(wk2gt_5,150,1010). coords(wk2gt_5,114,120,1010).  visible(wk2gt_5,1010). happensAt(walking(wk2gt_5),1010).
orientation(wk2gt_2,140,1011). coords(wk2gt_2,97,65,1011).  visible(wk2gt_2,1011). happensAt(inactive(wk2gt_2),1011). orientation(wk2gt_5,150,1011). coords(wk2gt_5,114,121,1011).  visible(wk2gt_5,1011). happensAt(walking(wk2gt_5),1011).
orientation(wk2gt_2,140,1012). coords(wk2gt_2,97,65,1012).  visible(wk2gt_2,1012). happensAt(inactive(wk2gt_2),1012). orientation(wk2gt_5,150,1012). coords(wk2gt_5,116,122,1012).  visible(wk2gt_5,1012). happensAt(walking(wk2gt_5),1012).
orientation(wk2gt_2,140,1013). coords(wk2gt_2,97,65,1013).  visible(wk2gt_2,1013). happensAt(inactive(wk2gt_2),1013). orientation(wk2gt_5,150,1013). coords(wk2gt_5,118,122,1013).  visible(wk2gt_5,1013). happensAt(walking(wk2gt_5),1013).
orientation(wk2gt_2,140,1014). coords(wk2gt_2,97,65,1014).  visible(wk2gt_2,1014). happensAt(inactive(wk2gt_2),1014). orientation(wk2gt_5,150,1014). coords(wk2gt_5,119,123,1014).  visible(wk2gt_5,1014). happensAt(walking(wk2gt_5),1014).
orientation(wk2gt_2,140,1015). coords(wk2gt_2,97,65,1015).  visible(wk2gt_2,1015). happensAt(inactive(wk2gt_2),1015). orientation(wk2gt_5,150,1015). coords(wk2gt_5,120,123,1015).  visible(wk2gt_5,1015). happensAt(walking(wk2gt_5),1015).
orientation(wk2gt_2,140,1016). coords(wk2gt_2,97,65,1016).  visible(wk2gt_2,1016). happensAt(inactive(wk2gt_2),1016). orientation(wk2gt_5,150,1016). coords(wk2gt_5,122,124,1016).  visible(wk2gt_5,1016). happensAt(walking(wk2gt_5),1016).
#show holdsAt/2.