
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
orientation(lb2gt_5,152,1007). coords(lb2gt_5,137,144,1007).  visible(lb2gt_5,1007). happensAt(walking(lb2gt_5),1007). orientation(lb2gt_7,0,1007). coords(lb2gt_7,97,264,1007).  visible(lb2gt_7,1007). happensAt(walking(lb2gt_7),1007).
orientation(lb2gt_5,152,1008). coords(lb2gt_5,137,144,1008).  visible(lb2gt_5,1008). happensAt(walking(lb2gt_5),1008). orientation(lb2gt_7,0,1008). coords(lb2gt_7,97,264,1008).  visible(lb2gt_7,1008). happensAt(walking(lb2gt_7),1008).
orientation(lb2gt_5,152,1009). coords(lb2gt_5,138,145,1009).  visible(lb2gt_5,1009). happensAt(walking(lb2gt_5),1009). orientation(lb2gt_7,0,1009). coords(lb2gt_7,97,264,1009).  visible(lb2gt_7,1009). happensAt(walking(lb2gt_7),1009).
orientation(lb2gt_5,148,1010). coords(lb2gt_5,138,145,1010).  visible(lb2gt_5,1010). happensAt(walking(lb2gt_5),1010). orientation(lb2gt_7,0,1010). coords(lb2gt_7,96,264,1010).  visible(lb2gt_7,1010). happensAt(walking(lb2gt_7),1010).
orientation(lb2gt_5,148,1011). coords(lb2gt_5,139,145,1011).  visible(lb2gt_5,1011). happensAt(walking(lb2gt_5),1011). orientation(lb2gt_7,0,1011). coords(lb2gt_7,96,265,1011).  visible(lb2gt_7,1011). happensAt(walking(lb2gt_7),1011).
orientation(lb2gt_5,147,1012). coords(lb2gt_5,139,145,1012).  visible(lb2gt_5,1012). happensAt(walking(lb2gt_5),1012). orientation(lb2gt_7,0,1012). coords(lb2gt_7,96,265,1012).  visible(lb2gt_7,1012). happensAt(walking(lb2gt_7),1012).
orientation(lb2gt_5,146,1013). coords(lb2gt_5,139,145,1013).  visible(lb2gt_5,1013). happensAt(walking(lb2gt_5),1013). orientation(lb2gt_7,0,1013). coords(lb2gt_7,96,265,1013).  visible(lb2gt_7,1013). happensAt(walking(lb2gt_7),1013).
orientation(lb2gt_5,145,1014). coords(lb2gt_5,142,145,1014).  visible(lb2gt_5,1014). happensAt(walking(lb2gt_5),1014). orientation(lb2gt_7,0,1014). coords(lb2gt_7,95,266,1014).  visible(lb2gt_7,1014). happensAt(walking(lb2gt_7),1014).
orientation(lb2gt_5,148,1015). coords(lb2gt_5,144,146,1015).  visible(lb2gt_5,1015). happensAt(walking(lb2gt_5),1015). orientation(lb2gt_7,0,1015). coords(lb2gt_7,95,266,1015).  visible(lb2gt_7,1015). happensAt(walking(lb2gt_7),1015).
orientation(lb2gt_5,148,1016). coords(lb2gt_5,147,146,1016).  visible(lb2gt_5,1016). happensAt(walking(lb2gt_5),1016). orientation(lb2gt_7,0,1016). coords(lb2gt_7,95,267,1016).  visible(lb2gt_7,1016). happensAt(walking(lb2gt_7),1016).
orientation(lb2gt_5,146,1017). coords(lb2gt_5,149,147,1017).  visible(lb2gt_5,1017). happensAt(walking(lb2gt_5),1017). orientation(lb2gt_7,0,1017). coords(lb2gt_7,94,267,1017).  visible(lb2gt_7,1017). happensAt(walking(lb2gt_7),1017).
orientation(lb2gt_5,146,1018). coords(lb2gt_5,151,148,1018).  visible(lb2gt_5,1018). happensAt(walking(lb2gt_5),1018). orientation(lb2gt_7,0,1018). coords(lb2gt_7,94,268,1018).  visible(lb2gt_7,1018). happensAt(walking(lb2gt_7),1018).
#show holdsAt/2.