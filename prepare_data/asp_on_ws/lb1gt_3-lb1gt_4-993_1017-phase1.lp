
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
orientation(lb1gt_3,134,1006). coords(lb1gt_3,198,80,1006).  visible(lb1gt_3,1006). happensAt(walking(lb1gt_3),1006). orientation(lb1gt_4,90,1006). coords(lb1gt_4,310,112,1006).  visible(lb1gt_4,1006). happensAt(inactive(lb1gt_4),1006).
orientation(lb1gt_3,134,1007). coords(lb1gt_3,195,80,1007).  visible(lb1gt_3,1007). happensAt(walking(lb1gt_3),1007). orientation(lb1gt_4,90,1007). coords(lb1gt_4,310,112,1007).  visible(lb1gt_4,1007). happensAt(inactive(lb1gt_4),1007).
orientation(lb1gt_3,134,1008). coords(lb1gt_3,193,80,1008).  visible(lb1gt_3,1008). happensAt(walking(lb1gt_3),1008). orientation(lb1gt_4,90,1008). coords(lb1gt_4,310,112,1008).  visible(lb1gt_4,1008). happensAt(inactive(lb1gt_4),1008).
orientation(lb1gt_3,134,1009). coords(lb1gt_3,189,80,1009).  visible(lb1gt_3,1009). happensAt(walking(lb1gt_3),1009). orientation(lb1gt_4,90,1009). coords(lb1gt_4,310,112,1009).  visible(lb1gt_4,1009). happensAt(inactive(lb1gt_4),1009).
orientation(lb1gt_3,134,1010). coords(lb1gt_3,186,79,1010).  visible(lb1gt_3,1010). happensAt(walking(lb1gt_3),1010). orientation(lb1gt_4,90,1010). coords(lb1gt_4,310,112,1010).  visible(lb1gt_4,1010). happensAt(inactive(lb1gt_4),1010).
orientation(lb1gt_3,134,1011). coords(lb1gt_3,184,79,1011).  visible(lb1gt_3,1011). happensAt(walking(lb1gt_3),1011). orientation(lb1gt_4,90,1011). coords(lb1gt_4,310,112,1011).  visible(lb1gt_4,1011). happensAt(inactive(lb1gt_4),1011).
orientation(lb1gt_3,134,1012). coords(lb1gt_3,182,79,1012).  visible(lb1gt_3,1012). happensAt(walking(lb1gt_3),1012). orientation(lb1gt_4,90,1012). coords(lb1gt_4,310,112,1012).  visible(lb1gt_4,1012). happensAt(inactive(lb1gt_4),1012).
orientation(lb1gt_3,134,1013). coords(lb1gt_3,180,78,1013).  visible(lb1gt_3,1013). happensAt(walking(lb1gt_3),1013). orientation(lb1gt_4,90,1013). coords(lb1gt_4,310,112,1013).  visible(lb1gt_4,1013). happensAt(inactive(lb1gt_4),1013).
orientation(lb1gt_3,134,1014). coords(lb1gt_3,179,78,1014).  visible(lb1gt_3,1014). happensAt(walking(lb1gt_3),1014). orientation(lb1gt_4,90,1014). coords(lb1gt_4,310,112,1014).  visible(lb1gt_4,1014). happensAt(inactive(lb1gt_4),1014).
orientation(lb1gt_3,134,1015). coords(lb1gt_3,179,78,1015).  visible(lb1gt_3,1015). happensAt(walking(lb1gt_3),1015). orientation(lb1gt_4,90,1015). coords(lb1gt_4,310,112,1015).  visible(lb1gt_4,1015). happensAt(inactive(lb1gt_4),1015).
orientation(lb1gt_3,134,1016). coords(lb1gt_3,179,78,1016).  visible(lb1gt_3,1016). happensAt(walking(lb1gt_3),1016). orientation(lb1gt_4,90,1016). coords(lb1gt_4,310,112,1016).  visible(lb1gt_4,1016). happensAt(inactive(lb1gt_4),1016).
orientation(lb1gt_3,134,1017). coords(lb1gt_3,178,78,1017).  visible(lb1gt_3,1017). happensAt(walking(lb1gt_3),1017). orientation(lb1gt_4,90,1017). coords(lb1gt_4,310,112,1017).  visible(lb1gt_4,1017). happensAt(inactive(lb1gt_4),1017).
#show holdsAt/2.