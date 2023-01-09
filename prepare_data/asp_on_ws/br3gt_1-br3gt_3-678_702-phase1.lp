
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

	orientation(br3gt_1,131,678). coords(br3gt_1,186,47,678).  visible(br3gt_1,678). happensAt(walking(br3gt_1),678). orientation(br3gt_3,150,678). coords(br3gt_3,6,79,678).  visible(br3gt_3,678). happensAt(walking(br3gt_3),678).
orientation(br3gt_1,131,679). coords(br3gt_1,186,47,679).  visible(br3gt_1,679). happensAt(walking(br3gt_1),679). orientation(br3gt_3,150,679). coords(br3gt_3,6,79,679).  visible(br3gt_3,679). happensAt(walking(br3gt_3),679).
orientation(br3gt_1,136,680). coords(br3gt_1,188,47,680).  visible(br3gt_1,680). happensAt(walking(br3gt_1),680). orientation(br3gt_3,150,680). coords(br3gt_3,6,78,680).  visible(br3gt_3,680). happensAt(walking(br3gt_3),680).
orientation(br3gt_1,136,681). coords(br3gt_1,190,48,681).  visible(br3gt_1,681). happensAt(walking(br3gt_1),681). orientation(br3gt_3,150,681). coords(br3gt_3,6,78,681).  visible(br3gt_3,681). happensAt(walking(br3gt_3),681).
orientation(br3gt_1,136,682). coords(br3gt_1,191,48,682).  visible(br3gt_1,682). happensAt(walking(br3gt_1),682). orientation(br3gt_3,150,682). coords(br3gt_3,6,77,682).  visible(br3gt_3,682). happensAt(walking(br3gt_3),682).
orientation(br3gt_1,136,683). coords(br3gt_1,192,48,683).  visible(br3gt_1,683). happensAt(walking(br3gt_1),683). orientation(br3gt_3,150,683). coords(br3gt_3,6,75,683).  visible(br3gt_3,683). happensAt(walking(br3gt_3),683).
orientation(br3gt_1,138,684). coords(br3gt_1,192,50,684).  visible(br3gt_1,684). happensAt(walking(br3gt_1),684). orientation(br3gt_3,150,684). coords(br3gt_3,6,75,684).  visible(br3gt_3,684). happensAt(walking(br3gt_3),684).
orientation(br3gt_1,138,685). coords(br3gt_1,192,50,685).  visible(br3gt_1,685). happensAt(walking(br3gt_1),685). orientation(br3gt_3,150,685). coords(br3gt_3,6,75,685).  visible(br3gt_3,685). happensAt(walking(br3gt_3),685).
orientation(br3gt_1,138,686). coords(br3gt_1,192,51,686).  visible(br3gt_1,686). happensAt(walking(br3gt_1),686). orientation(br3gt_3,150,686). coords(br3gt_3,7,73,686).  visible(br3gt_3,686). happensAt(walking(br3gt_3),686).
orientation(br3gt_1,138,687). coords(br3gt_1,193,51,687).  visible(br3gt_1,687). happensAt(walking(br3gt_1),687). orientation(br3gt_3,150,687). coords(br3gt_3,7,72,687).  visible(br3gt_3,687). happensAt(walking(br3gt_3),687).
orientation(br3gt_1,138,688). coords(br3gt_1,193,52,688).  visible(br3gt_1,688). happensAt(walking(br3gt_1),688). orientation(br3gt_3,150,688). coords(br3gt_3,7,72,688).  visible(br3gt_3,688). happensAt(walking(br3gt_3),688).
orientation(br3gt_1,138,689). coords(br3gt_1,193,52,689).  visible(br3gt_1,689). happensAt(walking(br3gt_1),689). orientation(br3gt_3,150,689). coords(br3gt_3,7,72,689).  visible(br3gt_3,689). happensAt(walking(br3gt_3),689).
orientation(br3gt_1,138,690). coords(br3gt_1,194,52,690).  visible(br3gt_1,690). happensAt(walking(br3gt_1),690). orientation(br3gt_3,150,690). coords(br3gt_3,8,70,690).  visible(br3gt_3,690). happensAt(walking(br3gt_3),690).
orientation(br3gt_1,135,691). coords(br3gt_1,194,52,691).  visible(br3gt_1,691). happensAt(walking(br3gt_1),691). orientation(br3gt_3,150,691). coords(br3gt_3,8,69,691).  visible(br3gt_3,691). happensAt(walking(br3gt_3),691).
orientation(br3gt_1,135,692). coords(br3gt_1,194,52,692).  visible(br3gt_1,692). happensAt(walking(br3gt_1),692). orientation(br3gt_3,150,692). coords(br3gt_3,8,68,692).  visible(br3gt_3,692). happensAt(walking(br3gt_3),692).
orientation(br3gt_1,135,693). coords(br3gt_1,194,52,693).  visible(br3gt_1,693). happensAt(walking(br3gt_1),693). orientation(br3gt_3,150,693). coords(br3gt_3,8,67,693).  visible(br3gt_3,693). happensAt(walking(br3gt_3),693).
orientation(br3gt_1,135,694). coords(br3gt_1,194,52,694).  visible(br3gt_1,694). happensAt(walking(br3gt_1),694). orientation(br3gt_3,150,694). coords(br3gt_3,8,67,694).  visible(br3gt_3,694). happensAt(walking(br3gt_3),694).
orientation(br3gt_1,135,695). coords(br3gt_1,195,53,695).  visible(br3gt_1,695). happensAt(walking(br3gt_1),695). orientation(br3gt_3,147,695). coords(br3gt_3,9,65,695).  visible(br3gt_3,695). happensAt(walking(br3gt_3),695).
orientation(br3gt_1,135,696). coords(br3gt_1,195,55,696).  visible(br3gt_1,696). happensAt(walking(br3gt_1),696). orientation(br3gt_3,147,696). coords(br3gt_3,9,64,696).  visible(br3gt_3,696). happensAt(walking(br3gt_3),696).
orientation(br3gt_1,135,697). coords(br3gt_1,195,57,697).  visible(br3gt_1,697). happensAt(walking(br3gt_1),697). orientation(br3gt_3,147,697). coords(br3gt_3,9,64,697).  visible(br3gt_3,697). happensAt(walking(br3gt_3),697).
orientation(br3gt_1,135,698). coords(br3gt_1,196,58,698).  visible(br3gt_1,698). happensAt(walking(br3gt_1),698). orientation(br3gt_3,147,698). coords(br3gt_3,9,63,698).  visible(br3gt_3,698). happensAt(walking(br3gt_3),698).
orientation(br3gt_1,135,699). coords(br3gt_1,196,59,699).  visible(br3gt_1,699). happensAt(walking(br3gt_1),699). orientation(br3gt_3,147,699). coords(br3gt_3,10,63,699).  visible(br3gt_3,699). happensAt(walking(br3gt_3),699).
orientation(br3gt_1,135,700). coords(br3gt_1,196,59,700).  visible(br3gt_1,700). happensAt(walking(br3gt_1),700). orientation(br3gt_3,147,700). coords(br3gt_3,10,63,700).  visible(br3gt_3,700). happensAt(walking(br3gt_3),700).
orientation(br3gt_1,135,701). coords(br3gt_1,197,60,701).  visible(br3gt_1,701). happensAt(walking(br3gt_1),701). orientation(br3gt_3,147,701). coords(br3gt_3,10,62,701).  visible(br3gt_3,701). happensAt(walking(br3gt_3),701).
orientation(br3gt_1,135,702). coords(br3gt_1,197,61,702).  visible(br3gt_1,702). happensAt(walking(br3gt_1),702). orientation(br3gt_3,147,702). coords(br3gt_3,11,61,702).  visible(br3gt_3,702). happensAt(walking(br3gt_3),702).
#show holdsAt/2.