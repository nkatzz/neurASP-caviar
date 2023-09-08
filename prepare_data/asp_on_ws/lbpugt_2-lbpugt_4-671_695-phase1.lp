
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
orientation(lbpugt_2,131,684). coords(lbpugt_2,228,76,684).  visible(lbpugt_2,684). happensAt(walking(lbpugt_2),684). orientation(lbpugt_4,135,684). coords(lbpugt_4,219,34,684).  visible(lbpugt_4,684). happensAt(inactive(lbpugt_4),684).
orientation(lbpugt_2,131,685). coords(lbpugt_2,231,77,685).  visible(lbpugt_2,685). happensAt(walking(lbpugt_2),685). orientation(lbpugt_4,135,685). coords(lbpugt_4,219,34,685).  visible(lbpugt_4,685). happensAt(inactive(lbpugt_4),685).
orientation(lbpugt_2,137,686). coords(lbpugt_2,233,77,686).  visible(lbpugt_2,686). happensAt(walking(lbpugt_2),686). orientation(lbpugt_4,135,686). coords(lbpugt_4,219,34,686).  visible(lbpugt_4,686). happensAt(inactive(lbpugt_4),686).
orientation(lbpugt_2,132,687). coords(lbpugt_2,234,78,687).  visible(lbpugt_2,687). happensAt(walking(lbpugt_2),687). orientation(lbpugt_4,135,687). coords(lbpugt_4,219,34,687).  visible(lbpugt_4,687). happensAt(inactive(lbpugt_4),687).
orientation(lbpugt_2,132,688). coords(lbpugt_2,237,79,688).  visible(lbpugt_2,688). happensAt(walking(lbpugt_2),688). orientation(lbpugt_4,135,688). coords(lbpugt_4,219,34,688).  visible(lbpugt_4,688). happensAt(inactive(lbpugt_4),688).
orientation(lbpugt_2,132,689). coords(lbpugt_2,238,79,689).  visible(lbpugt_2,689). happensAt(walking(lbpugt_2),689). orientation(lbpugt_4,135,689). coords(lbpugt_4,219,34,689).  visible(lbpugt_4,689). happensAt(inactive(lbpugt_4),689).
orientation(lbpugt_2,132,690). coords(lbpugt_2,239,79,690).  visible(lbpugt_2,690). happensAt(walking(lbpugt_2),690). orientation(lbpugt_4,135,690). coords(lbpugt_4,219,34,690).  visible(lbpugt_4,690). happensAt(inactive(lbpugt_4),690).
orientation(lbpugt_2,132,691). coords(lbpugt_2,240,79,691).  visible(lbpugt_2,691). happensAt(walking(lbpugt_2),691). orientation(lbpugt_4,135,691). coords(lbpugt_4,219,34,691).  visible(lbpugt_4,691). happensAt(inactive(lbpugt_4),691).
orientation(lbpugt_2,132,692). coords(lbpugt_2,241,80,692).  visible(lbpugt_2,692). happensAt(walking(lbpugt_2),692). orientation(lbpugt_4,135,692). coords(lbpugt_4,219,34,692).  visible(lbpugt_4,692). happensAt(inactive(lbpugt_4),692).
orientation(lbpugt_2,126,693). coords(lbpugt_2,241,80,693).  visible(lbpugt_2,693). happensAt(walking(lbpugt_2),693). orientation(lbpugt_4,135,693). coords(lbpugt_4,219,34,693).  visible(lbpugt_4,693). happensAt(inactive(lbpugt_4),693).
orientation(lbpugt_2,126,694). coords(lbpugt_2,242,80,694).  visible(lbpugt_2,694). happensAt(walking(lbpugt_2),694). orientation(lbpugt_4,135,694). coords(lbpugt_4,219,34,694).  visible(lbpugt_4,694). happensAt(inactive(lbpugt_4),694).
orientation(lbpugt_2,124,695). coords(lbpugt_2,243,80,695).  visible(lbpugt_2,695). happensAt(walking(lbpugt_2),695). orientation(lbpugt_4,135,695). coords(lbpugt_4,219,34,695).  visible(lbpugt_4,695). happensAt(inactive(lbpugt_4),695).
#show holdsAt/2.