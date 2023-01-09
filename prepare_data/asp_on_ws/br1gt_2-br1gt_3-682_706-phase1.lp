
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
orientation(br1gt_2,12,695). coords(br1gt_2,86,209,695).  visible(br1gt_2,695). happensAt(inactive(br1gt_2),695). orientation(br1gt_3,96,695). coords(br1gt_3,297,99,695).  visible(br1gt_3,695). happensAt(active(br1gt_3),695).
orientation(br1gt_2,12,696). coords(br1gt_2,86,209,696).  visible(br1gt_2,696). happensAt(inactive(br1gt_2),696). orientation(br1gt_3,96,696). coords(br1gt_3,297,99,696).  visible(br1gt_3,696). happensAt(active(br1gt_3),696).
orientation(br1gt_2,12,697). coords(br1gt_2,86,209,697).  visible(br1gt_2,697). happensAt(inactive(br1gt_2),697). orientation(br1gt_3,96,697). coords(br1gt_3,297,99,697).  visible(br1gt_3,697). happensAt(active(br1gt_3),697).
orientation(br1gt_2,12,698). coords(br1gt_2,86,209,698).  visible(br1gt_2,698). happensAt(inactive(br1gt_2),698). orientation(br1gt_3,96,698). coords(br1gt_3,297,99,698).  visible(br1gt_3,698). happensAt(active(br1gt_3),698).
orientation(br1gt_2,12,699). coords(br1gt_2,86,209,699).  visible(br1gt_2,699). happensAt(inactive(br1gt_2),699). orientation(br1gt_3,96,699). coords(br1gt_3,297,98,699).  visible(br1gt_3,699). happensAt(active(br1gt_3),699).
orientation(br1gt_2,12,700). coords(br1gt_2,86,209,700).  visible(br1gt_2,700). happensAt(inactive(br1gt_2),700). orientation(br1gt_3,96,700). coords(br1gt_3,297,98,700).  visible(br1gt_3,700). happensAt(active(br1gt_3),700).
orientation(br1gt_2,12,701). coords(br1gt_2,86,209,701).  visible(br1gt_2,701). happensAt(inactive(br1gt_2),701). orientation(br1gt_3,96,701). coords(br1gt_3,297,98,701).  visible(br1gt_3,701). happensAt(active(br1gt_3),701).
orientation(br1gt_2,12,702). coords(br1gt_2,86,209,702).  visible(br1gt_2,702). happensAt(inactive(br1gt_2),702). orientation(br1gt_3,96,702). coords(br1gt_3,297,98,702).  visible(br1gt_3,702). happensAt(active(br1gt_3),702).
orientation(br1gt_2,12,703). coords(br1gt_2,86,209,703).  visible(br1gt_2,703). happensAt(inactive(br1gt_2),703). orientation(br1gt_3,103,703). coords(br1gt_3,299,97,703).  visible(br1gt_3,703). happensAt(active(br1gt_3),703).
orientation(br1gt_2,12,704). coords(br1gt_2,86,209,704).  visible(br1gt_2,704). happensAt(inactive(br1gt_2),704). orientation(br1gt_3,103,704). coords(br1gt_3,299,97,704).  visible(br1gt_3,704). happensAt(active(br1gt_3),704).
orientation(br1gt_2,12,705). coords(br1gt_2,86,209,705).  visible(br1gt_2,705). happensAt(inactive(br1gt_2),705). orientation(br1gt_3,103,705). coords(br1gt_3,299,96,705).  visible(br1gt_3,705). happensAt(active(br1gt_3),705).
orientation(br1gt_2,12,706). coords(br1gt_2,86,209,706).  visible(br1gt_2,706). happensAt(inactive(br1gt_2),706). orientation(br1gt_3,103,706). coords(br1gt_3,299,96,706).  visible(br1gt_3,706). happensAt(active(br1gt_3),706).
#show holdsAt/2.