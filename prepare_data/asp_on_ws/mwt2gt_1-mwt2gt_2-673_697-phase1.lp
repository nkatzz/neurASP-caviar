
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

	orientation(mwt2gt_1,153,673). coords(mwt2gt_1,26,131,673).  visible(mwt2gt_1,673). happensAt(inactive(mwt2gt_1),673). orientation(mwt2gt_2,152,673). coords(mwt2gt_2,24,138,673).  visible(mwt2gt_2,673). happensAt(inactive(mwt2gt_2),673).
orientation(mwt2gt_1,153,674). coords(mwt2gt_1,26,131,674).  visible(mwt2gt_1,674). happensAt(inactive(mwt2gt_1),674). orientation(mwt2gt_2,152,674). coords(mwt2gt_2,24,138,674).  visible(mwt2gt_2,674). happensAt(inactive(mwt2gt_2),674).
orientation(mwt2gt_1,153,675). coords(mwt2gt_1,26,131,675).  visible(mwt2gt_1,675). happensAt(inactive(mwt2gt_1),675). orientation(mwt2gt_2,152,675). coords(mwt2gt_2,24,138,675).  visible(mwt2gt_2,675). happensAt(inactive(mwt2gt_2),675).
orientation(mwt2gt_1,153,676). coords(mwt2gt_1,26,131,676).  visible(mwt2gt_1,676). happensAt(inactive(mwt2gt_1),676). orientation(mwt2gt_2,152,676). coords(mwt2gt_2,24,138,676).  visible(mwt2gt_2,676). happensAt(inactive(mwt2gt_2),676).
orientation(mwt2gt_1,153,677). coords(mwt2gt_1,26,131,677).  visible(mwt2gt_1,677). happensAt(inactive(mwt2gt_1),677). orientation(mwt2gt_2,152,677). coords(mwt2gt_2,24,138,677).  visible(mwt2gt_2,677). happensAt(inactive(mwt2gt_2),677).
orientation(mwt2gt_1,153,678). coords(mwt2gt_1,26,131,678).  visible(mwt2gt_1,678). happensAt(inactive(mwt2gt_1),678). orientation(mwt2gt_2,152,678). coords(mwt2gt_2,24,138,678).  visible(mwt2gt_2,678). happensAt(inactive(mwt2gt_2),678).
orientation(mwt2gt_1,153,679). coords(mwt2gt_1,26,131,679).  visible(mwt2gt_1,679). happensAt(inactive(mwt2gt_1),679). orientation(mwt2gt_2,152,679). coords(mwt2gt_2,24,138,679).  visible(mwt2gt_2,679). happensAt(inactive(mwt2gt_2),679).
orientation(mwt2gt_1,153,680). coords(mwt2gt_1,26,131,680).  visible(mwt2gt_1,680). happensAt(inactive(mwt2gt_1),680). orientation(mwt2gt_2,152,680). coords(mwt2gt_2,24,138,680).  visible(mwt2gt_2,680). happensAt(inactive(mwt2gt_2),680).
orientation(mwt2gt_1,153,681). coords(mwt2gt_1,26,130,681).  visible(mwt2gt_1,681). happensAt(inactive(mwt2gt_1),681). orientation(mwt2gt_2,152,681). coords(mwt2gt_2,24,138,681).  visible(mwt2gt_2,681). happensAt(inactive(mwt2gt_2),681).
orientation(mwt2gt_1,153,682). coords(mwt2gt_1,25,130,682).  visible(mwt2gt_1,682). happensAt(inactive(mwt2gt_1),682). orientation(mwt2gt_2,152,682). coords(mwt2gt_2,24,138,682).  visible(mwt2gt_2,682). happensAt(inactive(mwt2gt_2),682).
orientation(mwt2gt_1,153,683). coords(mwt2gt_1,26,131,683).  visible(mwt2gt_1,683). happensAt(inactive(mwt2gt_1),683). orientation(mwt2gt_2,152,683). coords(mwt2gt_2,24,138,683).  visible(mwt2gt_2,683). happensAt(inactive(mwt2gt_2),683).
orientation(mwt2gt_1,153,684). coords(mwt2gt_1,26,130,684).  visible(mwt2gt_1,684). happensAt(inactive(mwt2gt_1),684). orientation(mwt2gt_2,152,684). coords(mwt2gt_2,24,138,684).  visible(mwt2gt_2,684). happensAt(inactive(mwt2gt_2),684).
orientation(mwt2gt_1,153,685). coords(mwt2gt_1,26,131,685).  visible(mwt2gt_1,685). happensAt(inactive(mwt2gt_1),685). orientation(mwt2gt_2,152,685). coords(mwt2gt_2,24,138,685).  visible(mwt2gt_2,685). happensAt(inactive(mwt2gt_2),685).
orientation(mwt2gt_1,153,686). coords(mwt2gt_1,26,130,686).  visible(mwt2gt_1,686). happensAt(inactive(mwt2gt_1),686). orientation(mwt2gt_2,152,686). coords(mwt2gt_2,24,138,686).  visible(mwt2gt_2,686). happensAt(inactive(mwt2gt_2),686).
orientation(mwt2gt_1,153,687). coords(mwt2gt_1,26,130,687).  visible(mwt2gt_1,687). happensAt(inactive(mwt2gt_1),687). orientation(mwt2gt_2,152,687). coords(mwt2gt_2,24,138,687).  visible(mwt2gt_2,687). happensAt(inactive(mwt2gt_2),687).
orientation(mwt2gt_1,153,688). coords(mwt2gt_1,26,130,688).  visible(mwt2gt_1,688). happensAt(inactive(mwt2gt_1),688). orientation(mwt2gt_2,152,688). coords(mwt2gt_2,24,138,688).  visible(mwt2gt_2,688). happensAt(inactive(mwt2gt_2),688).
orientation(mwt2gt_1,153,689). coords(mwt2gt_1,26,130,689).  visible(mwt2gt_1,689). happensAt(inactive(mwt2gt_1),689). orientation(mwt2gt_2,152,689). coords(mwt2gt_2,24,138,689).  visible(mwt2gt_2,689). happensAt(inactive(mwt2gt_2),689).
orientation(mwt2gt_1,153,690). coords(mwt2gt_1,26,130,690).  visible(mwt2gt_1,690). happensAt(inactive(mwt2gt_1),690). orientation(mwt2gt_2,152,690). coords(mwt2gt_2,24,138,690).  visible(mwt2gt_2,690). happensAt(inactive(mwt2gt_2),690).
orientation(mwt2gt_1,153,691). coords(mwt2gt_1,26,130,691).  visible(mwt2gt_1,691). happensAt(inactive(mwt2gt_1),691). orientation(mwt2gt_2,152,691). coords(mwt2gt_2,24,138,691).  visible(mwt2gt_2,691). happensAt(inactive(mwt2gt_2),691).
orientation(mwt2gt_1,153,692). coords(mwt2gt_1,26,130,692).  visible(mwt2gt_1,692). happensAt(inactive(mwt2gt_1),692). orientation(mwt2gt_2,152,692). coords(mwt2gt_2,24,138,692).  visible(mwt2gt_2,692). happensAt(inactive(mwt2gt_2),692).
orientation(mwt2gt_1,153,693). coords(mwt2gt_1,26,129,693).  visible(mwt2gt_1,693). happensAt(inactive(mwt2gt_1),693). orientation(mwt2gt_2,152,693). coords(mwt2gt_2,24,138,693).  visible(mwt2gt_2,693). happensAt(inactive(mwt2gt_2),693).
orientation(mwt2gt_1,153,694). coords(mwt2gt_1,26,129,694).  visible(mwt2gt_1,694). happensAt(inactive(mwt2gt_1),694). orientation(mwt2gt_2,152,694). coords(mwt2gt_2,24,138,694).  visible(mwt2gt_2,694). happensAt(inactive(mwt2gt_2),694).
orientation(mwt2gt_1,153,695). coords(mwt2gt_1,26,129,695).  visible(mwt2gt_1,695). happensAt(inactive(mwt2gt_1),695). orientation(mwt2gt_2,152,695). coords(mwt2gt_2,24,138,695).  visible(mwt2gt_2,695). happensAt(inactive(mwt2gt_2),695).
orientation(mwt2gt_1,153,696). coords(mwt2gt_1,26,129,696).  visible(mwt2gt_1,696). happensAt(inactive(mwt2gt_1),696). orientation(mwt2gt_2,152,696). coords(mwt2gt_2,24,138,696).  visible(mwt2gt_2,696). happensAt(inactive(mwt2gt_2),696).
orientation(mwt2gt_1,153,697). coords(mwt2gt_1,26,129,697).  visible(mwt2gt_1,697). happensAt(inactive(mwt2gt_1),697). orientation(mwt2gt_2,152,697). coords(mwt2gt_2,24,138,697).  visible(mwt2gt_2,697). happensAt(inactive(mwt2gt_2),697).
#show holdsAt/2.