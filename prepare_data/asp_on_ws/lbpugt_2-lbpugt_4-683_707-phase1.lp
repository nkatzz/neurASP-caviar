
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
orientation(lbpugt_2,124,696). coords(lbpugt_2,243,81,696).  visible(lbpugt_2,696). happensAt(walking(lbpugt_2),696). orientation(lbpugt_4,135,696). coords(lbpugt_4,219,34,696).  visible(lbpugt_4,696). happensAt(inactive(lbpugt_4),696).
orientation(lbpugt_2,124,697). coords(lbpugt_2,245,82,697).  visible(lbpugt_2,697). happensAt(walking(lbpugt_2),697). orientation(lbpugt_4,135,697). coords(lbpugt_4,219,34,697).  visible(lbpugt_4,697). happensAt(inactive(lbpugt_4),697).
orientation(lbpugt_2,118,698). coords(lbpugt_2,248,83,698).  visible(lbpugt_2,698). happensAt(walking(lbpugt_2),698). orientation(lbpugt_4,135,698). coords(lbpugt_4,219,34,698).  visible(lbpugt_4,698). happensAt(inactive(lbpugt_4),698).
orientation(lbpugt_2,122,699). coords(lbpugt_2,251,85,699).  visible(lbpugt_2,699). happensAt(walking(lbpugt_2),699). orientation(lbpugt_4,135,699). coords(lbpugt_4,219,34,699).  visible(lbpugt_4,699). happensAt(inactive(lbpugt_4),699).
orientation(lbpugt_2,122,700). coords(lbpugt_2,255,86,700).  visible(lbpugt_2,700). happensAt(walking(lbpugt_2),700). orientation(lbpugt_4,135,700). coords(lbpugt_4,219,34,700).  visible(lbpugt_4,700). happensAt(inactive(lbpugt_4),700).
orientation(lbpugt_2,122,701). coords(lbpugt_2,258,86,701).  visible(lbpugt_2,701). happensAt(walking(lbpugt_2),701). orientation(lbpugt_4,135,701). coords(lbpugt_4,219,34,701).  visible(lbpugt_4,701). happensAt(inactive(lbpugt_4),701).
orientation(lbpugt_2,122,702). coords(lbpugt_2,259,87,702).  visible(lbpugt_2,702). happensAt(walking(lbpugt_2),702). orientation(lbpugt_4,135,702). coords(lbpugt_4,219,34,702).  visible(lbpugt_4,702). happensAt(inactive(lbpugt_4),702).
orientation(lbpugt_2,131,703). coords(lbpugt_2,261,87,703).  visible(lbpugt_2,703). happensAt(walking(lbpugt_2),703). orientation(lbpugt_4,135,703). coords(lbpugt_4,219,34,703).  visible(lbpugt_4,703). happensAt(inactive(lbpugt_4),703).
orientation(lbpugt_2,131,704). coords(lbpugt_2,262,87,704).  visible(lbpugt_2,704). happensAt(walking(lbpugt_2),704). orientation(lbpugt_4,135,704). coords(lbpugt_4,219,34,704).  visible(lbpugt_4,704). happensAt(inactive(lbpugt_4),704).
orientation(lbpugt_2,131,705). coords(lbpugt_2,263,88,705).  visible(lbpugt_2,705). happensAt(walking(lbpugt_2),705). orientation(lbpugt_4,135,705). coords(lbpugt_4,219,34,705).  visible(lbpugt_4,705). happensAt(inactive(lbpugt_4),705).
orientation(lbpugt_2,131,706). coords(lbpugt_2,264,88,706).  visible(lbpugt_2,706). happensAt(walking(lbpugt_2),706). orientation(lbpugt_4,135,706). coords(lbpugt_4,219,34,706).  visible(lbpugt_4,706). happensAt(inactive(lbpugt_4),706).
orientation(lbpugt_2,126,707). coords(lbpugt_2,265,88,707).  visible(lbpugt_2,707). happensAt(walking(lbpugt_2),707). orientation(lbpugt_4,135,707). coords(lbpugt_4,219,34,707).  visible(lbpugt_4,707). happensAt(inactive(lbpugt_4),707).
#show holdsAt/2.