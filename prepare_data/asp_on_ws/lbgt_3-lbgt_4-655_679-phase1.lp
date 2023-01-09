
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

	orientation(lbgt_3,61,655). coords(lbgt_3,146,162,655).  visible(lbgt_3,655). happensAt(inactive(lbgt_3),655). orientation(lbgt_4,161,655). coords(lbgt_4,125,186,655).  visible(lbgt_4,655). happensAt(walking(lbgt_4),655).
orientation(lbgt_3,61,656). coords(lbgt_3,146,162,656).  visible(lbgt_3,656). happensAt(inactive(lbgt_3),656). orientation(lbgt_4,161,656). coords(lbgt_4,125,186,656).  visible(lbgt_4,656). happensAt(walking(lbgt_4),656).
orientation(lbgt_3,61,657). coords(lbgt_3,146,162,657).  visible(lbgt_3,657). happensAt(inactive(lbgt_3),657). orientation(lbgt_4,161,657). coords(lbgt_4,125,186,657).  visible(lbgt_4,657). happensAt(walking(lbgt_4),657).
orientation(lbgt_3,61,658). coords(lbgt_3,146,162,658).  visible(lbgt_3,658). happensAt(inactive(lbgt_3),658). orientation(lbgt_4,161,658). coords(lbgt_4,127,187,658).  visible(lbgt_4,658). happensAt(walking(lbgt_4),658).
orientation(lbgt_3,61,659). coords(lbgt_3,146,162,659).  visible(lbgt_3,659). happensAt(inactive(lbgt_3),659). orientation(lbgt_4,161,659). coords(lbgt_4,127,187,659).  visible(lbgt_4,659). happensAt(walking(lbgt_4),659).
orientation(lbgt_3,61,660). coords(lbgt_3,146,162,660).  visible(lbgt_3,660). happensAt(inactive(lbgt_3),660). orientation(lbgt_4,161,660). coords(lbgt_4,127,187,660).  visible(lbgt_4,660). happensAt(walking(lbgt_4),660).
orientation(lbgt_3,61,661). coords(lbgt_3,146,162,661).  visible(lbgt_3,661). happensAt(inactive(lbgt_3),661). orientation(lbgt_4,161,661). coords(lbgt_4,127,187,661).  visible(lbgt_4,661). happensAt(walking(lbgt_4),661).
orientation(lbgt_3,61,662). coords(lbgt_3,146,162,662).  visible(lbgt_3,662). happensAt(inactive(lbgt_3),662). orientation(lbgt_4,161,662). coords(lbgt_4,127,187,662).  visible(lbgt_4,662). happensAt(walking(lbgt_4),662).
orientation(lbgt_3,61,663). coords(lbgt_3,146,162,663).  visible(lbgt_3,663). happensAt(inactive(lbgt_3),663). orientation(lbgt_4,161,663). coords(lbgt_4,127,187,663).  visible(lbgt_4,663). happensAt(walking(lbgt_4),663).
orientation(lbgt_3,61,664). coords(lbgt_3,146,162,664).  visible(lbgt_3,664). happensAt(inactive(lbgt_3),664). orientation(lbgt_4,161,664). coords(lbgt_4,128,187,664).  visible(lbgt_4,664). happensAt(walking(lbgt_4),664).
orientation(lbgt_3,61,665). coords(lbgt_3,146,162,665).  visible(lbgt_3,665). happensAt(inactive(lbgt_3),665). orientation(lbgt_4,161,665). coords(lbgt_4,128,187,665).  visible(lbgt_4,665). happensAt(walking(lbgt_4),665).
orientation(lbgt_3,61,666). coords(lbgt_3,146,162,666).  visible(lbgt_3,666). happensAt(inactive(lbgt_3),666). orientation(lbgt_4,161,666). coords(lbgt_4,130,188,666).  visible(lbgt_4,666). happensAt(active(lbgt_4),666).
orientation(lbgt_3,61,667). coords(lbgt_3,146,162,667).  visible(lbgt_3,667). happensAt(inactive(lbgt_3),667). orientation(lbgt_4,161,667). coords(lbgt_4,130,188,667).  visible(lbgt_4,667). happensAt(active(lbgt_4),667).
orientation(lbgt_3,61,668). coords(lbgt_3,146,162,668).  visible(lbgt_3,668). happensAt(inactive(lbgt_3),668). orientation(lbgt_4,161,668). coords(lbgt_4,130,188,668).  visible(lbgt_4,668). happensAt(active(lbgt_4),668).
orientation(lbgt_3,61,669). coords(lbgt_3,146,162,669).  visible(lbgt_3,669). happensAt(inactive(lbgt_3),669). orientation(lbgt_4,161,669). coords(lbgt_4,130,189,669).  visible(lbgt_4,669). happensAt(active(lbgt_4),669).
orientation(lbgt_3,61,670). coords(lbgt_3,146,162,670).  visible(lbgt_3,670). happensAt(inactive(lbgt_3),670). orientation(lbgt_4,161,670). coords(lbgt_4,130,189,670).  visible(lbgt_4,670). happensAt(active(lbgt_4),670).
orientation(lbgt_3,61,671). coords(lbgt_3,146,162,671).  visible(lbgt_3,671). happensAt(inactive(lbgt_3),671). orientation(lbgt_4,161,671). coords(lbgt_4,130,189,671).  visible(lbgt_4,671). happensAt(active(lbgt_4),671).
orientation(lbgt_3,61,672). coords(lbgt_3,146,162,672).  visible(lbgt_3,672). happensAt(inactive(lbgt_3),672). orientation(lbgt_4,161,672). coords(lbgt_4,130,189,672).  visible(lbgt_4,672). happensAt(active(lbgt_4),672).
orientation(lbgt_3,61,673). coords(lbgt_3,146,162,673).  visible(lbgt_3,673). happensAt(inactive(lbgt_3),673). orientation(lbgt_4,161,673). coords(lbgt_4,130,189,673).  visible(lbgt_4,673). happensAt(active(lbgt_4),673).
orientation(lbgt_3,61,674). coords(lbgt_3,146,162,674).  visible(lbgt_3,674). happensAt(inactive(lbgt_3),674). orientation(lbgt_4,161,674). coords(lbgt_4,130,189,674).  visible(lbgt_4,674). happensAt(active(lbgt_4),674).
orientation(lbgt_3,61,675). coords(lbgt_3,146,162,675).  visible(lbgt_3,675). happensAt(inactive(lbgt_3),675). orientation(lbgt_4,161,675). coords(lbgt_4,130,189,675).  visible(lbgt_4,675). happensAt(active(lbgt_4),675).
orientation(lbgt_3,61,676). coords(lbgt_3,146,162,676).  visible(lbgt_3,676). happensAt(inactive(lbgt_3),676). orientation(lbgt_4,161,676). coords(lbgt_4,130,189,676).  visible(lbgt_4,676). happensAt(active(lbgt_4),676).
orientation(lbgt_3,61,677). coords(lbgt_3,146,162,677).  visible(lbgt_3,677). happensAt(inactive(lbgt_3),677). orientation(lbgt_4,161,677). coords(lbgt_4,130,189,677).  visible(lbgt_4,677). happensAt(active(lbgt_4),677).
orientation(lbgt_3,61,678). coords(lbgt_3,146,162,678).  visible(lbgt_3,678). happensAt(inactive(lbgt_3),678). orientation(lbgt_4,161,678). coords(lbgt_4,130,189,678).  visible(lbgt_4,678). happensAt(active(lbgt_4),678).
orientation(lbgt_3,61,679). coords(lbgt_3,146,162,679).  visible(lbgt_3,679). happensAt(inactive(lbgt_3),679). orientation(lbgt_4,161,679). coords(lbgt_4,130,188,679).  visible(lbgt_4,679). happensAt(active(lbgt_4),679).
#show holdsAt/2.