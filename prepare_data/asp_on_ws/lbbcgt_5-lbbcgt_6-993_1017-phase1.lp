
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

	orientation(lbbcgt_5,154,993). coords(lbbcgt_5,180,118,993).  visible(lbbcgt_5,993). happensAt(active(lbbcgt_5),993). orientation(lbbcgt_6,151,993). coords(lbbcgt_6,69,91,993).  visible(lbbcgt_6,993). happensAt(walking(lbbcgt_6),993).
orientation(lbbcgt_5,154,994). coords(lbbcgt_5,180,118,994).  visible(lbbcgt_5,994). happensAt(active(lbbcgt_5),994). orientation(lbbcgt_6,151,994). coords(lbbcgt_6,69,91,994).  visible(lbbcgt_6,994). happensAt(walking(lbbcgt_6),994).
orientation(lbbcgt_5,154,995). coords(lbbcgt_5,180,118,995).  visible(lbbcgt_5,995). happensAt(active(lbbcgt_5),995). orientation(lbbcgt_6,151,995). coords(lbbcgt_6,68,91,995).  visible(lbbcgt_6,995). happensAt(walking(lbbcgt_6),995).
orientation(lbbcgt_5,154,996). coords(lbbcgt_5,180,117,996).  visible(lbbcgt_5,996). happensAt(active(lbbcgt_5),996). orientation(lbbcgt_6,151,996). coords(lbbcgt_6,67,91,996).  visible(lbbcgt_6,996). happensAt(walking(lbbcgt_6),996).
orientation(lbbcgt_5,154,997). coords(lbbcgt_5,179,117,997).  visible(lbbcgt_5,997). happensAt(active(lbbcgt_5),997). orientation(lbbcgt_6,151,997). coords(lbbcgt_6,67,91,997).  visible(lbbcgt_6,997). happensAt(walking(lbbcgt_6),997).
orientation(lbbcgt_5,154,998). coords(lbbcgt_5,178,117,998).  visible(lbbcgt_5,998). happensAt(active(lbbcgt_5),998). orientation(lbbcgt_6,151,998). coords(lbbcgt_6,66,92,998).  visible(lbbcgt_6,998). happensAt(walking(lbbcgt_6),998).
orientation(lbbcgt_5,154,999). coords(lbbcgt_5,179,117,999).  visible(lbbcgt_5,999). happensAt(active(lbbcgt_5),999). orientation(lbbcgt_6,151,999). coords(lbbcgt_6,65,92,999).  visible(lbbcgt_6,999). happensAt(walking(lbbcgt_6),999).
orientation(lbbcgt_5,154,1000). coords(lbbcgt_5,180,117,1000).  visible(lbbcgt_5,1000). happensAt(active(lbbcgt_5),1000). orientation(lbbcgt_6,151,1000). coords(lbbcgt_6,65,93,1000).  visible(lbbcgt_6,1000). happensAt(walking(lbbcgt_6),1000).
orientation(lbbcgt_5,154,1001). coords(lbbcgt_5,180,117,1001).  visible(lbbcgt_5,1001). happensAt(active(lbbcgt_5),1001). orientation(lbbcgt_6,151,1001). coords(lbbcgt_6,65,93,1001).  visible(lbbcgt_6,1001). happensAt(walking(lbbcgt_6),1001).
orientation(lbbcgt_5,154,1002). coords(lbbcgt_5,180,117,1002).  visible(lbbcgt_5,1002). happensAt(active(lbbcgt_5),1002). orientation(lbbcgt_6,151,1002). coords(lbbcgt_6,65,94,1002).  visible(lbbcgt_6,1002). happensAt(walking(lbbcgt_6),1002).
orientation(lbbcgt_5,154,1003). coords(lbbcgt_5,179,117,1003).  visible(lbbcgt_5,1003). happensAt(active(lbbcgt_5),1003). orientation(lbbcgt_6,151,1003). coords(lbbcgt_6,65,94,1003).  visible(lbbcgt_6,1003). happensAt(walking(lbbcgt_6),1003).
orientation(lbbcgt_5,154,1004). coords(lbbcgt_5,179,117,1004).  visible(lbbcgt_5,1004). happensAt(active(lbbcgt_5),1004). orientation(lbbcgt_6,151,1004). coords(lbbcgt_6,65,94,1004).  visible(lbbcgt_6,1004). happensAt(walking(lbbcgt_6),1004).
orientation(lbbcgt_5,154,1005). coords(lbbcgt_5,179,117,1005).  visible(lbbcgt_5,1005). happensAt(active(lbbcgt_5),1005). orientation(lbbcgt_6,151,1005). coords(lbbcgt_6,64,94,1005).  visible(lbbcgt_6,1005). happensAt(walking(lbbcgt_6),1005).
orientation(lbbcgt_5,154,1006). coords(lbbcgt_5,178,117,1006).  visible(lbbcgt_5,1006). happensAt(active(lbbcgt_5),1006). orientation(lbbcgt_6,151,1006). coords(lbbcgt_6,65,94,1006).  visible(lbbcgt_6,1006). happensAt(walking(lbbcgt_6),1006).
orientation(lbbcgt_5,154,1007). coords(lbbcgt_5,178,117,1007).  visible(lbbcgt_5,1007). happensAt(active(lbbcgt_5),1007). orientation(lbbcgt_6,151,1007). coords(lbbcgt_6,65,94,1007).  visible(lbbcgt_6,1007). happensAt(walking(lbbcgt_6),1007).
orientation(lbbcgt_5,154,1008). coords(lbbcgt_5,178,117,1008).  visible(lbbcgt_5,1008). happensAt(active(lbbcgt_5),1008). orientation(lbbcgt_6,151,1008). coords(lbbcgt_6,64,94,1008).  visible(lbbcgt_6,1008). happensAt(walking(lbbcgt_6),1008).
orientation(lbbcgt_5,154,1009). coords(lbbcgt_5,178,117,1009).  visible(lbbcgt_5,1009). happensAt(active(lbbcgt_5),1009). orientation(lbbcgt_6,151,1009). coords(lbbcgt_6,63,94,1009).  visible(lbbcgt_6,1009). happensAt(walking(lbbcgt_6),1009).
orientation(lbbcgt_5,154,1010). coords(lbbcgt_5,178,117,1010).  visible(lbbcgt_5,1010). happensAt(active(lbbcgt_5),1010). orientation(lbbcgt_6,151,1010). coords(lbbcgt_6,63,94,1010).  visible(lbbcgt_6,1010). happensAt(walking(lbbcgt_6),1010).
orientation(lbbcgt_5,154,1011). coords(lbbcgt_5,178,117,1011).  visible(lbbcgt_5,1011). happensAt(active(lbbcgt_5),1011). orientation(lbbcgt_6,151,1011). coords(lbbcgt_6,63,94,1011).  visible(lbbcgt_6,1011). happensAt(walking(lbbcgt_6),1011).
orientation(lbbcgt_5,154,1012). coords(lbbcgt_5,178,117,1012).  visible(lbbcgt_5,1012). happensAt(active(lbbcgt_5),1012). orientation(lbbcgt_6,151,1012). coords(lbbcgt_6,62,94,1012).  visible(lbbcgt_6,1012). happensAt(walking(lbbcgt_6),1012).
orientation(lbbcgt_5,154,1013). coords(lbbcgt_5,179,117,1013).  visible(lbbcgt_5,1013). happensAt(active(lbbcgt_5),1013). orientation(lbbcgt_6,151,1013). coords(lbbcgt_6,63,94,1013).  visible(lbbcgt_6,1013). happensAt(walking(lbbcgt_6),1013).
orientation(lbbcgt_5,154,1014). coords(lbbcgt_5,178,117,1014).  visible(lbbcgt_5,1014). happensAt(active(lbbcgt_5),1014). orientation(lbbcgt_6,151,1014). coords(lbbcgt_6,63,94,1014).  visible(lbbcgt_6,1014). happensAt(walking(lbbcgt_6),1014).
orientation(lbbcgt_5,154,1015). coords(lbbcgt_5,178,117,1015).  visible(lbbcgt_5,1015). happensAt(active(lbbcgt_5),1015). orientation(lbbcgt_6,151,1015). coords(lbbcgt_6,63,94,1015).  visible(lbbcgt_6,1015). happensAt(walking(lbbcgt_6),1015).
orientation(lbbcgt_5,154,1016). coords(lbbcgt_5,178,117,1016).  visible(lbbcgt_5,1016). happensAt(active(lbbcgt_5),1016). orientation(lbbcgt_6,151,1016). coords(lbbcgt_6,63,94,1016).  visible(lbbcgt_6,1016). happensAt(walking(lbbcgt_6),1016).
orientation(lbbcgt_5,154,1017). coords(lbbcgt_5,178,117,1017).  visible(lbbcgt_5,1017). happensAt(active(lbbcgt_5),1017). orientation(lbbcgt_6,151,1017). coords(lbbcgt_6,62,94,1017).  visible(lbbcgt_6,1017). happensAt(walking(lbbcgt_6),1017).
#show holdsAt/2.