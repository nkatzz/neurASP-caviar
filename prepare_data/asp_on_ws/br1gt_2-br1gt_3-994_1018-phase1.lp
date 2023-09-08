
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

	orientation(br1gt_2,174,994). coords(br1gt_2,108,209,994).  visible(br1gt_2,994). happensAt(walking(br1gt_2),994). orientation(br1gt_3,123,994). coords(br1gt_3,278,95,994).  visible(br1gt_3,994). happensAt(walking(br1gt_3),994).
orientation(br1gt_2,174,995). coords(br1gt_2,109,209,995).  visible(br1gt_2,995). happensAt(walking(br1gt_2),995). orientation(br1gt_3,123,995). coords(br1gt_3,276,95,995).  visible(br1gt_3,995). happensAt(walking(br1gt_3),995).
orientation(br1gt_2,174,996). coords(br1gt_2,109,209,996).  visible(br1gt_2,996). happensAt(walking(br1gt_2),996). orientation(br1gt_3,123,996). coords(br1gt_3,275,95,996).  visible(br1gt_3,996). happensAt(walking(br1gt_3),996).
orientation(br1gt_2,174,997). coords(br1gt_2,109,210,997).  visible(br1gt_2,997). happensAt(walking(br1gt_2),997). orientation(br1gt_3,123,997). coords(br1gt_3,273,95,997).  visible(br1gt_3,997). happensAt(walking(br1gt_3),997).
orientation(br1gt_2,174,998). coords(br1gt_2,109,210,998).  visible(br1gt_2,998). happensAt(walking(br1gt_2),998). orientation(br1gt_3,123,998). coords(br1gt_3,270,96,998).  visible(br1gt_3,998). happensAt(walking(br1gt_3),998).
orientation(br1gt_2,174,999). coords(br1gt_2,109,211,999).  visible(br1gt_2,999). happensAt(walking(br1gt_2),999). orientation(br1gt_3,123,999). coords(br1gt_3,268,96,999).  visible(br1gt_3,999). happensAt(walking(br1gt_3),999).
orientation(br1gt_2,174,1000). coords(br1gt_2,109,211,1000).  visible(br1gt_2,1000). happensAt(walking(br1gt_2),1000). orientation(br1gt_3,123,1000). coords(br1gt_3,266,96,1000).  visible(br1gt_3,1000). happensAt(walking(br1gt_3),1000).
orientation(br1gt_2,174,1001). coords(br1gt_2,109,212,1001).  visible(br1gt_2,1001). happensAt(walking(br1gt_2),1001). orientation(br1gt_3,123,1001). coords(br1gt_3,263,96,1001).  visible(br1gt_3,1001). happensAt(walking(br1gt_3),1001).
orientation(br1gt_2,174,1002). coords(br1gt_2,110,212,1002).  visible(br1gt_2,1002). happensAt(walking(br1gt_2),1002). orientation(br1gt_3,123,1002). coords(br1gt_3,260,96,1002).  visible(br1gt_3,1002). happensAt(walking(br1gt_3),1002).
orientation(br1gt_2,174,1003). coords(br1gt_2,110,212,1003).  visible(br1gt_2,1003). happensAt(walking(br1gt_2),1003). orientation(br1gt_3,123,1003). coords(br1gt_3,258,97,1003).  visible(br1gt_3,1003). happensAt(walking(br1gt_3),1003).
orientation(br1gt_2,174,1004). coords(br1gt_2,111,213,1004).  visible(br1gt_2,1004). happensAt(walking(br1gt_2),1004). orientation(br1gt_3,123,1004). coords(br1gt_3,258,97,1004).  visible(br1gt_3,1004). happensAt(walking(br1gt_3),1004).
orientation(br1gt_2,174,1005). coords(br1gt_2,110,214,1005).  visible(br1gt_2,1005). happensAt(walking(br1gt_2),1005). orientation(br1gt_3,128,1005). coords(br1gt_3,257,97,1005).  visible(br1gt_3,1005). happensAt(walking(br1gt_3),1005).
orientation(br1gt_2,174,1006). coords(br1gt_2,110,214,1006).  visible(br1gt_2,1006). happensAt(walking(br1gt_2),1006). orientation(br1gt_3,128,1006). coords(br1gt_3,256,97,1006).  visible(br1gt_3,1006). happensAt(walking(br1gt_3),1006).
orientation(br1gt_2,174,1007). coords(br1gt_2,111,214,1007).  visible(br1gt_2,1007). happensAt(walking(br1gt_2),1007). orientation(br1gt_3,128,1007). coords(br1gt_3,256,97,1007).  visible(br1gt_3,1007). happensAt(walking(br1gt_3),1007).
orientation(br1gt_2,174,1008). coords(br1gt_2,110,213,1008).  visible(br1gt_2,1008). happensAt(walking(br1gt_2),1008). orientation(br1gt_3,128,1008). coords(br1gt_3,254,97,1008).  visible(br1gt_3,1008). happensAt(walking(br1gt_3),1008).
orientation(br1gt_2,174,1009). coords(br1gt_2,110,213,1009).  visible(br1gt_2,1009). happensAt(walking(br1gt_2),1009). orientation(br1gt_3,128,1009). coords(br1gt_3,253,97,1009).  visible(br1gt_3,1009). happensAt(walking(br1gt_3),1009).
orientation(br1gt_2,174,1010). coords(br1gt_2,111,215,1010).  visible(br1gt_2,1010). happensAt(walking(br1gt_2),1010). orientation(br1gt_3,135,1010). coords(br1gt_3,252,97,1010).  visible(br1gt_3,1010). happensAt(walking(br1gt_3),1010).
orientation(br1gt_2,174,1011). coords(br1gt_2,111,215,1011).  visible(br1gt_2,1011). happensAt(walking(br1gt_2),1011). orientation(br1gt_3,135,1011). coords(br1gt_3,250,96,1011).  visible(br1gt_3,1011). happensAt(walking(br1gt_3),1011).
orientation(br1gt_2,174,1012). coords(br1gt_2,111,215,1012).  visible(br1gt_2,1012). happensAt(walking(br1gt_2),1012). orientation(br1gt_3,135,1012). coords(br1gt_3,249,96,1012).  visible(br1gt_3,1012). happensAt(walking(br1gt_3),1012).
orientation(br1gt_2,174,1013). coords(br1gt_2,111,216,1013).  visible(br1gt_2,1013). happensAt(walking(br1gt_2),1013). orientation(br1gt_3,135,1013). coords(br1gt_3,245,96,1013).  visible(br1gt_3,1013). happensAt(walking(br1gt_3),1013).
orientation(br1gt_2,174,1014). coords(br1gt_2,111,216,1014).  visible(br1gt_2,1014). happensAt(walking(br1gt_2),1014). orientation(br1gt_3,135,1014). coords(br1gt_3,243,94,1014).  visible(br1gt_3,1014). happensAt(walking(br1gt_3),1014).
orientation(br1gt_2,174,1015). coords(br1gt_2,111,215,1015).  visible(br1gt_2,1015). happensAt(walking(br1gt_2),1015). orientation(br1gt_3,135,1015). coords(br1gt_3,240,94,1015).  visible(br1gt_3,1015). happensAt(walking(br1gt_3),1015).
orientation(br1gt_2,174,1016). coords(br1gt_2,111,216,1016).  visible(br1gt_2,1016). happensAt(walking(br1gt_2),1016). orientation(br1gt_3,135,1016). coords(br1gt_3,237,95,1016).  visible(br1gt_3,1016). happensAt(walking(br1gt_3),1016).
orientation(br1gt_2,174,1017). coords(br1gt_2,111,214,1017).  visible(br1gt_2,1017). happensAt(walking(br1gt_2),1017). orientation(br1gt_3,135,1017). coords(br1gt_3,235,95,1017).  visible(br1gt_3,1017). happensAt(walking(br1gt_3),1017).
orientation(br1gt_2,174,1018). coords(br1gt_2,111,214,1018).  visible(br1gt_2,1018). happensAt(walking(br1gt_2),1018). orientation(br1gt_3,135,1018). coords(br1gt_3,233,95,1018).  visible(br1gt_3,1018). happensAt(walking(br1gt_3),1018).
#show holdsAt/2.