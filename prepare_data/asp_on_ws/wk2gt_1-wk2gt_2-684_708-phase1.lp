
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

	orientation(wk2gt_1,125,684). coords(wk2gt_1,114,53,684).  visible(wk2gt_1,684). happensAt(inactive(wk2gt_1),684). orientation(wk2gt_2,142,684). coords(wk2gt_2,99,63,684).  visible(wk2gt_2,684). happensAt(inactive(wk2gt_2),684).
orientation(wk2gt_1,125,685). coords(wk2gt_1,114,53,685).  visible(wk2gt_1,685). happensAt(inactive(wk2gt_1),685). orientation(wk2gt_2,142,685). coords(wk2gt_2,99,63,685).  visible(wk2gt_2,685). happensAt(inactive(wk2gt_2),685).
orientation(wk2gt_1,125,686). coords(wk2gt_1,114,53,686).  visible(wk2gt_1,686). happensAt(inactive(wk2gt_1),686). orientation(wk2gt_2,142,686). coords(wk2gt_2,99,63,686).  visible(wk2gt_2,686). happensAt(inactive(wk2gt_2),686).
orientation(wk2gt_1,125,687). coords(wk2gt_1,114,53,687).  visible(wk2gt_1,687). happensAt(inactive(wk2gt_1),687). orientation(wk2gt_2,142,687). coords(wk2gt_2,99,63,687).  visible(wk2gt_2,687). happensAt(inactive(wk2gt_2),687).
orientation(wk2gt_1,125,688). coords(wk2gt_1,114,53,688).  visible(wk2gt_1,688). happensAt(inactive(wk2gt_1),688). orientation(wk2gt_2,142,688). coords(wk2gt_2,99,63,688).  visible(wk2gt_2,688). happensAt(inactive(wk2gt_2),688).
orientation(wk2gt_1,125,689). coords(wk2gt_1,114,53,689).  visible(wk2gt_1,689). happensAt(inactive(wk2gt_1),689). orientation(wk2gt_2,142,689). coords(wk2gt_2,99,63,689).  visible(wk2gt_2,689). happensAt(inactive(wk2gt_2),689).
orientation(wk2gt_1,125,690). coords(wk2gt_1,114,53,690).  visible(wk2gt_1,690). happensAt(inactive(wk2gt_1),690). orientation(wk2gt_2,142,690). coords(wk2gt_2,99,63,690).  visible(wk2gt_2,690). happensAt(inactive(wk2gt_2),690).
orientation(wk2gt_1,125,691). coords(wk2gt_1,114,53,691).  visible(wk2gt_1,691). happensAt(inactive(wk2gt_1),691). orientation(wk2gt_2,142,691). coords(wk2gt_2,99,63,691).  visible(wk2gt_2,691). happensAt(inactive(wk2gt_2),691).
orientation(wk2gt_1,125,692). coords(wk2gt_1,114,53,692).  visible(wk2gt_1,692). happensAt(inactive(wk2gt_1),692). orientation(wk2gt_2,142,692). coords(wk2gt_2,99,63,692).  visible(wk2gt_2,692). happensAt(inactive(wk2gt_2),692).
orientation(wk2gt_1,125,693). coords(wk2gt_1,114,53,693).  visible(wk2gt_1,693). happensAt(inactive(wk2gt_1),693). orientation(wk2gt_2,142,693). coords(wk2gt_2,99,63,693).  visible(wk2gt_2,693). happensAt(inactive(wk2gt_2),693).
orientation(wk2gt_1,125,694). coords(wk2gt_1,114,53,694).  visible(wk2gt_1,694). happensAt(inactive(wk2gt_1),694). orientation(wk2gt_2,142,694). coords(wk2gt_2,99,63,694).  visible(wk2gt_2,694). happensAt(inactive(wk2gt_2),694).
orientation(wk2gt_1,125,695). coords(wk2gt_1,114,53,695).  visible(wk2gt_1,695). happensAt(inactive(wk2gt_1),695). orientation(wk2gt_2,142,695). coords(wk2gt_2,99,63,695).  visible(wk2gt_2,695). happensAt(inactive(wk2gt_2),695).
orientation(wk2gt_1,125,696). coords(wk2gt_1,114,53,696).  visible(wk2gt_1,696). happensAt(inactive(wk2gt_1),696). orientation(wk2gt_2,142,696). coords(wk2gt_2,99,63,696).  visible(wk2gt_2,696). happensAt(inactive(wk2gt_2),696).
orientation(wk2gt_1,125,697). coords(wk2gt_1,114,53,697).  visible(wk2gt_1,697). happensAt(inactive(wk2gt_1),697). orientation(wk2gt_2,142,697). coords(wk2gt_2,99,63,697).  visible(wk2gt_2,697). happensAt(inactive(wk2gt_2),697).
orientation(wk2gt_1,125,698). coords(wk2gt_1,114,53,698).  visible(wk2gt_1,698). happensAt(inactive(wk2gt_1),698). orientation(wk2gt_2,142,698). coords(wk2gt_2,99,63,698).  visible(wk2gt_2,698). happensAt(inactive(wk2gt_2),698).
orientation(wk2gt_1,125,699). coords(wk2gt_1,114,53,699).  visible(wk2gt_1,699). happensAt(inactive(wk2gt_1),699). orientation(wk2gt_2,142,699). coords(wk2gt_2,99,63,699).  visible(wk2gt_2,699). happensAt(inactive(wk2gt_2),699).
orientation(wk2gt_1,125,700). coords(wk2gt_1,114,53,700).  visible(wk2gt_1,700). happensAt(inactive(wk2gt_1),700). orientation(wk2gt_2,142,700). coords(wk2gt_2,99,63,700).  visible(wk2gt_2,700). happensAt(inactive(wk2gt_2),700).
orientation(wk2gt_1,125,701). coords(wk2gt_1,114,53,701).  visible(wk2gt_1,701). happensAt(inactive(wk2gt_1),701). orientation(wk2gt_2,142,701). coords(wk2gt_2,99,63,701).  visible(wk2gt_2,701). happensAt(inactive(wk2gt_2),701).
orientation(wk2gt_1,125,702). coords(wk2gt_1,114,53,702).  visible(wk2gt_1,702). happensAt(inactive(wk2gt_1),702). orientation(wk2gt_2,142,702). coords(wk2gt_2,99,63,702).  visible(wk2gt_2,702). happensAt(inactive(wk2gt_2),702).
orientation(wk2gt_1,125,703). coords(wk2gt_1,114,53,703).  visible(wk2gt_1,703). happensAt(inactive(wk2gt_1),703). orientation(wk2gt_2,142,703). coords(wk2gt_2,99,63,703).  visible(wk2gt_2,703). happensAt(inactive(wk2gt_2),703).
orientation(wk2gt_1,125,704). coords(wk2gt_1,114,53,704).  visible(wk2gt_1,704). happensAt(inactive(wk2gt_1),704). orientation(wk2gt_2,142,704). coords(wk2gt_2,99,63,704).  visible(wk2gt_2,704). happensAt(inactive(wk2gt_2),704).
orientation(wk2gt_1,125,705). coords(wk2gt_1,114,53,705).  visible(wk2gt_1,705). happensAt(inactive(wk2gt_1),705). orientation(wk2gt_2,142,705). coords(wk2gt_2,99,63,705).  visible(wk2gt_2,705). happensAt(inactive(wk2gt_2),705).
orientation(wk2gt_1,125,706). coords(wk2gt_1,114,53,706).  visible(wk2gt_1,706). happensAt(inactive(wk2gt_1),706). orientation(wk2gt_2,142,706). coords(wk2gt_2,99,63,706).  visible(wk2gt_2,706). happensAt(inactive(wk2gt_2),706).
orientation(wk2gt_1,125,707). coords(wk2gt_1,114,53,707).  visible(wk2gt_1,707). happensAt(inactive(wk2gt_1),707). orientation(wk2gt_2,142,707). coords(wk2gt_2,99,63,707).  visible(wk2gt_2,707). happensAt(inactive(wk2gt_2),707).
orientation(wk2gt_1,125,708). coords(wk2gt_1,114,53,708).  visible(wk2gt_1,708). happensAt(inactive(wk2gt_1),708). orientation(wk2gt_2,142,708). coords(wk2gt_2,99,63,708).  visible(wk2gt_2,708). happensAt(inactive(wk2gt_2),708).
#show holdsAt/2.