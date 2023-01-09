
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

	orientation(br1gt_2,12,670). coords(br1gt_2,86,209,670).  visible(br1gt_2,670). happensAt(inactive(br1gt_2),670). orientation(br1gt_3,114,670). coords(br1gt_3,302,93,670).  visible(br1gt_3,670). happensAt(active(br1gt_3),670).
orientation(br1gt_2,12,671). coords(br1gt_2,86,209,671).  visible(br1gt_2,671). happensAt(inactive(br1gt_2),671). orientation(br1gt_3,105,671). coords(br1gt_3,302,96,671).  visible(br1gt_3,671). happensAt(active(br1gt_3),671).
orientation(br1gt_2,12,672). coords(br1gt_2,86,209,672).  visible(br1gt_2,672). happensAt(inactive(br1gt_2),672). orientation(br1gt_3,105,672). coords(br1gt_3,300,97,672).  visible(br1gt_3,672). happensAt(active(br1gt_3),672).
orientation(br1gt_2,12,673). coords(br1gt_2,86,209,673).  visible(br1gt_2,673). happensAt(inactive(br1gt_2),673). orientation(br1gt_3,105,673). coords(br1gt_3,300,97,673).  visible(br1gt_3,673). happensAt(active(br1gt_3),673).
orientation(br1gt_2,12,674). coords(br1gt_2,86,209,674).  visible(br1gt_2,674). happensAt(inactive(br1gt_2),674). orientation(br1gt_3,105,674). coords(br1gt_3,299,98,674).  visible(br1gt_3,674). happensAt(active(br1gt_3),674).
orientation(br1gt_2,12,675). coords(br1gt_2,86,209,675).  visible(br1gt_2,675). happensAt(inactive(br1gt_2),675). orientation(br1gt_3,102,675). coords(br1gt_3,299,98,675).  visible(br1gt_3,675). happensAt(active(br1gt_3),675).
orientation(br1gt_2,12,676). coords(br1gt_2,86,209,676).  visible(br1gt_2,676). happensAt(inactive(br1gt_2),676). orientation(br1gt_3,102,676). coords(br1gt_3,299,98,676).  visible(br1gt_3,676). happensAt(active(br1gt_3),676).
orientation(br1gt_2,12,677). coords(br1gt_2,86,209,677).  visible(br1gt_2,677). happensAt(inactive(br1gt_2),677). orientation(br1gt_3,102,677). coords(br1gt_3,296,98,677).  visible(br1gt_3,677). happensAt(active(br1gt_3),677).
orientation(br1gt_2,12,678). coords(br1gt_2,86,209,678).  visible(br1gt_2,678). happensAt(inactive(br1gt_2),678). orientation(br1gt_3,102,678). coords(br1gt_3,296,98,678).  visible(br1gt_3,678). happensAt(active(br1gt_3),678).
orientation(br1gt_2,12,679). coords(br1gt_2,86,209,679).  visible(br1gt_2,679). happensAt(inactive(br1gt_2),679). orientation(br1gt_3,102,679). coords(br1gt_3,296,98,679).  visible(br1gt_3,679). happensAt(active(br1gt_3),679).
orientation(br1gt_2,12,680). coords(br1gt_2,86,209,680).  visible(br1gt_2,680). happensAt(inactive(br1gt_2),680). orientation(br1gt_3,102,680). coords(br1gt_3,295,97,680).  visible(br1gt_3,680). happensAt(active(br1gt_3),680).
orientation(br1gt_2,12,681). coords(br1gt_2,86,209,681).  visible(br1gt_2,681). happensAt(inactive(br1gt_2),681). orientation(br1gt_3,102,681). coords(br1gt_3,295,97,681).  visible(br1gt_3,681). happensAt(active(br1gt_3),681).
orientation(br1gt_2,12,682). coords(br1gt_2,86,209,682).  visible(br1gt_2,682). happensAt(inactive(br1gt_2),682). orientation(br1gt_3,96,682). coords(br1gt_3,295,97,682).  visible(br1gt_3,682). happensAt(active(br1gt_3),682).
orientation(br1gt_2,12,683). coords(br1gt_2,86,209,683).  visible(br1gt_2,683). happensAt(inactive(br1gt_2),683). orientation(br1gt_3,96,683). coords(br1gt_3,295,97,683).  visible(br1gt_3,683). happensAt(active(br1gt_3),683).
orientation(br1gt_2,12,684). coords(br1gt_2,86,209,684).  visible(br1gt_2,684). happensAt(inactive(br1gt_2),684). orientation(br1gt_3,96,684). coords(br1gt_3,295,98,684).  visible(br1gt_3,684). happensAt(active(br1gt_3),684).
orientation(br1gt_2,12,685). coords(br1gt_2,86,209,685).  visible(br1gt_2,685). happensAt(inactive(br1gt_2),685). orientation(br1gt_3,96,685). coords(br1gt_3,295,98,685).  visible(br1gt_3,685). happensAt(active(br1gt_3),685).
orientation(br1gt_2,12,686). coords(br1gt_2,86,209,686).  visible(br1gt_2,686). happensAt(inactive(br1gt_2),686). orientation(br1gt_3,96,686). coords(br1gt_3,296,98,686).  visible(br1gt_3,686). happensAt(active(br1gt_3),686).
orientation(br1gt_2,12,687). coords(br1gt_2,86,209,687).  visible(br1gt_2,687). happensAt(inactive(br1gt_2),687). orientation(br1gt_3,96,687). coords(br1gt_3,296,98,687).  visible(br1gt_3,687). happensAt(active(br1gt_3),687).
orientation(br1gt_2,12,688). coords(br1gt_2,86,209,688).  visible(br1gt_2,688). happensAt(inactive(br1gt_2),688). orientation(br1gt_3,96,688). coords(br1gt_3,296,98,688).  visible(br1gt_3,688). happensAt(active(br1gt_3),688).
orientation(br1gt_2,12,689). coords(br1gt_2,86,209,689).  visible(br1gt_2,689). happensAt(inactive(br1gt_2),689). orientation(br1gt_3,96,689). coords(br1gt_3,296,98,689).  visible(br1gt_3,689). happensAt(active(br1gt_3),689).
orientation(br1gt_2,12,690). coords(br1gt_2,86,209,690).  visible(br1gt_2,690). happensAt(inactive(br1gt_2),690). orientation(br1gt_3,96,690). coords(br1gt_3,296,98,690).  visible(br1gt_3,690). happensAt(active(br1gt_3),690).
orientation(br1gt_2,12,691). coords(br1gt_2,86,209,691).  visible(br1gt_2,691). happensAt(inactive(br1gt_2),691). orientation(br1gt_3,96,691). coords(br1gt_3,296,98,691).  visible(br1gt_3,691). happensAt(active(br1gt_3),691).
orientation(br1gt_2,12,692). coords(br1gt_2,86,209,692).  visible(br1gt_2,692). happensAt(inactive(br1gt_2),692). orientation(br1gt_3,96,692). coords(br1gt_3,296,98,692).  visible(br1gt_3,692). happensAt(active(br1gt_3),692).
orientation(br1gt_2,12,693). coords(br1gt_2,86,209,693).  visible(br1gt_2,693). happensAt(inactive(br1gt_2),693). orientation(br1gt_3,96,693). coords(br1gt_3,296,99,693).  visible(br1gt_3,693). happensAt(active(br1gt_3),693).
orientation(br1gt_2,12,694). coords(br1gt_2,86,209,694).  visible(br1gt_2,694). happensAt(inactive(br1gt_2),694). orientation(br1gt_3,96,694). coords(br1gt_3,296,99,694).  visible(br1gt_3,694). happensAt(active(br1gt_3),694).
#show holdsAt/2.