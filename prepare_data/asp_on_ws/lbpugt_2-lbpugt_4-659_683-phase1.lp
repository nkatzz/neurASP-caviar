
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

	orientation(lbpugt_2,132,659). coords(lbpugt_2,197,63,659).  visible(lbpugt_2,659). happensAt(walking(lbpugt_2),659). orientation(lbpugt_4,135,659). coords(lbpugt_4,219,34,659).  visible(lbpugt_4,659). happensAt(inactive(lbpugt_4),659).
orientation(lbpugt_2,132,660). coords(lbpugt_2,198,64,660).  visible(lbpugt_2,660). happensAt(walking(lbpugt_2),660). orientation(lbpugt_4,135,660). coords(lbpugt_4,219,34,660).  visible(lbpugt_4,660). happensAt(inactive(lbpugt_4),660).
orientation(lbpugt_2,132,661). coords(lbpugt_2,198,64,661).  visible(lbpugt_2,661). happensAt(walking(lbpugt_2),661). orientation(lbpugt_4,135,661). coords(lbpugt_4,219,34,661).  visible(lbpugt_4,661). happensAt(inactive(lbpugt_4),661).
orientation(lbpugt_2,132,662). coords(lbpugt_2,199,64,662).  visible(lbpugt_2,662). happensAt(walking(lbpugt_2),662). orientation(lbpugt_4,135,662). coords(lbpugt_4,219,34,662).  visible(lbpugt_4,662). happensAt(inactive(lbpugt_4),662).
orientation(lbpugt_2,132,663). coords(lbpugt_2,199,64,663).  visible(lbpugt_2,663). happensAt(walking(lbpugt_2),663). orientation(lbpugt_4,135,663). coords(lbpugt_4,219,34,663).  visible(lbpugt_4,663). happensAt(inactive(lbpugt_4),663).
orientation(lbpugt_2,132,664). coords(lbpugt_2,199,64,664).  visible(lbpugt_2,664). happensAt(walking(lbpugt_2),664). orientation(lbpugt_4,135,664). coords(lbpugt_4,219,34,664).  visible(lbpugt_4,664). happensAt(inactive(lbpugt_4),664).
orientation(lbpugt_2,132,665). coords(lbpugt_2,200,64,665).  visible(lbpugt_2,665). happensAt(walking(lbpugt_2),665). orientation(lbpugt_4,135,665). coords(lbpugt_4,219,34,665).  visible(lbpugt_4,665). happensAt(inactive(lbpugt_4),665).
orientation(lbpugt_2,127,666). coords(lbpugt_2,200,64,666).  visible(lbpugt_2,666). happensAt(walking(lbpugt_2),666). orientation(lbpugt_4,135,666). coords(lbpugt_4,219,34,666).  visible(lbpugt_4,666). happensAt(inactive(lbpugt_4),666).
orientation(lbpugt_2,127,667). coords(lbpugt_2,200,65,667).  visible(lbpugt_2,667). happensAt(walking(lbpugt_2),667). orientation(lbpugt_4,135,667). coords(lbpugt_4,219,34,667).  visible(lbpugt_4,667). happensAt(inactive(lbpugt_4),667).
orientation(lbpugt_2,127,668). coords(lbpugt_2,202,66,668).  visible(lbpugt_2,668). happensAt(walking(lbpugt_2),668). orientation(lbpugt_4,135,668). coords(lbpugt_4,219,34,668).  visible(lbpugt_4,668). happensAt(inactive(lbpugt_4),668).
orientation(lbpugt_2,127,669). coords(lbpugt_2,204,67,669).  visible(lbpugt_2,669). happensAt(walking(lbpugt_2),669). orientation(lbpugt_4,135,669). coords(lbpugt_4,219,34,669).  visible(lbpugt_4,669). happensAt(inactive(lbpugt_4),669).
orientation(lbpugt_2,131,670). coords(lbpugt_2,207,68,670).  visible(lbpugt_2,670). happensAt(walking(lbpugt_2),670). orientation(lbpugt_4,135,670). coords(lbpugt_4,219,34,670).  visible(lbpugt_4,670). happensAt(inactive(lbpugt_4),670).
orientation(lbpugt_2,131,671). coords(lbpugt_2,210,70,671).  visible(lbpugt_2,671). happensAt(walking(lbpugt_2),671). orientation(lbpugt_4,135,671). coords(lbpugt_4,219,34,671).  visible(lbpugt_4,671). happensAt(inactive(lbpugt_4),671).
orientation(lbpugt_2,131,672). coords(lbpugt_2,211,71,672).  visible(lbpugt_2,672). happensAt(walking(lbpugt_2),672). orientation(lbpugt_4,135,672). coords(lbpugt_4,219,34,672).  visible(lbpugt_4,672). happensAt(inactive(lbpugt_4),672).
orientation(lbpugt_2,131,673). coords(lbpugt_2,212,72,673).  visible(lbpugt_2,673). happensAt(walking(lbpugt_2),673). orientation(lbpugt_4,135,673). coords(lbpugt_4,219,34,673).  visible(lbpugt_4,673). happensAt(inactive(lbpugt_4),673).
orientation(lbpugt_2,131,674). coords(lbpugt_2,215,73,674).  visible(lbpugt_2,674). happensAt(walking(lbpugt_2),674). orientation(lbpugt_4,135,674). coords(lbpugt_4,219,34,674).  visible(lbpugt_4,674). happensAt(inactive(lbpugt_4),674).
orientation(lbpugt_2,131,675). coords(lbpugt_2,216,73,675).  visible(lbpugt_2,675). happensAt(walking(lbpugt_2),675). orientation(lbpugt_4,135,675). coords(lbpugt_4,219,34,675).  visible(lbpugt_4,675). happensAt(inactive(lbpugt_4),675).
orientation(lbpugt_2,131,676). coords(lbpugt_2,217,74,676).  visible(lbpugt_2,676). happensAt(walking(lbpugt_2),676). orientation(lbpugt_4,135,676). coords(lbpugt_4,219,34,676).  visible(lbpugt_4,676). happensAt(inactive(lbpugt_4),676).
orientation(lbpugt_2,131,677). coords(lbpugt_2,217,74,677).  visible(lbpugt_2,677). happensAt(walking(lbpugt_2),677). orientation(lbpugt_4,135,677). coords(lbpugt_4,219,34,677).  visible(lbpugt_4,677). happensAt(inactive(lbpugt_4),677).
orientation(lbpugt_2,131,678). coords(lbpugt_2,218,74,678).  visible(lbpugt_2,678). happensAt(walking(lbpugt_2),678). orientation(lbpugt_4,135,678). coords(lbpugt_4,219,34,678).  visible(lbpugt_4,678). happensAt(inactive(lbpugt_4),678).
orientation(lbpugt_2,131,679). coords(lbpugt_2,218,74,679).  visible(lbpugt_2,679). happensAt(walking(lbpugt_2),679). orientation(lbpugt_4,135,679). coords(lbpugt_4,219,34,679).  visible(lbpugt_4,679). happensAt(inactive(lbpugt_4),679).
orientation(lbpugt_2,131,680). coords(lbpugt_2,219,74,680).  visible(lbpugt_2,680). happensAt(walking(lbpugt_2),680). orientation(lbpugt_4,135,680). coords(lbpugt_4,219,34,680).  visible(lbpugt_4,680). happensAt(inactive(lbpugt_4),680).
orientation(lbpugt_2,131,681). coords(lbpugt_2,219,74,681).  visible(lbpugt_2,681). happensAt(walking(lbpugt_2),681). orientation(lbpugt_4,135,681). coords(lbpugt_4,219,34,681).  visible(lbpugt_4,681). happensAt(inactive(lbpugt_4),681).
orientation(lbpugt_2,131,682). coords(lbpugt_2,222,74,682).  visible(lbpugt_2,682). happensAt(walking(lbpugt_2),682). orientation(lbpugt_4,135,682). coords(lbpugt_4,219,34,682).  visible(lbpugt_4,682). happensAt(inactive(lbpugt_4),682).
orientation(lbpugt_2,131,683). coords(lbpugt_2,226,75,683).  visible(lbpugt_2,683). happensAt(walking(lbpugt_2),683). orientation(lbpugt_4,135,683). coords(lbpugt_4,219,34,683).  visible(lbpugt_4,683). happensAt(inactive(lbpugt_4),683).
#show holdsAt/2.