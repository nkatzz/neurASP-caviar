
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

	orientation(lbgt_3,61,679). coords(lbgt_3,146,162,679).  visible(lbgt_3,679). happensAt(inactive(lbgt_3),679). orientation(lbgt_4,161,679). coords(lbgt_4,130,188,679).  visible(lbgt_4,679). happensAt(active(lbgt_4),679).
orientation(lbgt_3,61,680). coords(lbgt_3,146,162,680).  visible(lbgt_3,680). happensAt(inactive(lbgt_3),680). orientation(lbgt_4,161,680). coords(lbgt_4,130,188,680).  visible(lbgt_4,680). happensAt(active(lbgt_4),680).
orientation(lbgt_3,61,681). coords(lbgt_3,146,162,681).  visible(lbgt_3,681). happensAt(inactive(lbgt_3),681). orientation(lbgt_4,161,681). coords(lbgt_4,129,188,681).  visible(lbgt_4,681). happensAt(active(lbgt_4),681).
orientation(lbgt_3,61,682). coords(lbgt_3,146,162,682).  visible(lbgt_3,682). happensAt(inactive(lbgt_3),682). orientation(lbgt_4,161,682). coords(lbgt_4,129,188,682).  visible(lbgt_4,682). happensAt(active(lbgt_4),682).
orientation(lbgt_3,61,683). coords(lbgt_3,146,162,683).  visible(lbgt_3,683). happensAt(inactive(lbgt_3),683). orientation(lbgt_4,161,683). coords(lbgt_4,129,188,683).  visible(lbgt_4,683). happensAt(walking(lbgt_4),683).
orientation(lbgt_3,61,684). coords(lbgt_3,146,162,684).  visible(lbgt_3,684). happensAt(inactive(lbgt_3),684). orientation(lbgt_4,161,684). coords(lbgt_4,128,187,684).  visible(lbgt_4,684). happensAt(walking(lbgt_4),684).
orientation(lbgt_3,61,685). coords(lbgt_3,146,162,685).  visible(lbgt_3,685). happensAt(inactive(lbgt_3),685). orientation(lbgt_4,161,685). coords(lbgt_4,128,187,685).  visible(lbgt_4,685). happensAt(walking(lbgt_4),685).
orientation(lbgt_3,61,686). coords(lbgt_3,146,162,686).  visible(lbgt_3,686). happensAt(inactive(lbgt_3),686). orientation(lbgt_4,161,686). coords(lbgt_4,128,187,686).  visible(lbgt_4,686). happensAt(walking(lbgt_4),686).
orientation(lbgt_3,61,687). coords(lbgt_3,146,162,687).  visible(lbgt_3,687). happensAt(inactive(lbgt_3),687). orientation(lbgt_4,161,687). coords(lbgt_4,128,186,687).  visible(lbgt_4,687). happensAt(walking(lbgt_4),687).
orientation(lbgt_3,61,688). coords(lbgt_3,146,162,688).  visible(lbgt_3,688). happensAt(inactive(lbgt_3),688). orientation(lbgt_4,161,688). coords(lbgt_4,126,185,688).  visible(lbgt_4,688). happensAt(walking(lbgt_4),688).
orientation(lbgt_3,61,689). coords(lbgt_3,146,162,689).  visible(lbgt_3,689). happensAt(inactive(lbgt_3),689). orientation(lbgt_4,160,689). coords(lbgt_4,125,185,689).  visible(lbgt_4,689). happensAt(walking(lbgt_4),689).
orientation(lbgt_3,61,690). coords(lbgt_3,146,162,690).  visible(lbgt_3,690). happensAt(inactive(lbgt_3),690). orientation(lbgt_4,158,690). coords(lbgt_4,123,185,690).  visible(lbgt_4,690). happensAt(walking(lbgt_4),690).
orientation(lbgt_3,61,691). coords(lbgt_3,146,162,691).  visible(lbgt_3,691). happensAt(inactive(lbgt_3),691). orientation(lbgt_4,158,691). coords(lbgt_4,122,184,691).  visible(lbgt_4,691). happensAt(walking(lbgt_4),691).
orientation(lbgt_3,61,692). coords(lbgt_3,146,162,692).  visible(lbgt_3,692). happensAt(inactive(lbgt_3),692). orientation(lbgt_4,155,692). coords(lbgt_4,122,184,692).  visible(lbgt_4,692). happensAt(walking(lbgt_4),692).
orientation(lbgt_3,61,693). coords(lbgt_3,146,162,693).  visible(lbgt_3,693). happensAt(inactive(lbgt_3),693). orientation(lbgt_4,155,693). coords(lbgt_4,122,183,693).  visible(lbgt_4,693). happensAt(walking(lbgt_4),693).
orientation(lbgt_3,61,694). coords(lbgt_3,146,162,694).  visible(lbgt_3,694). happensAt(inactive(lbgt_3),694). orientation(lbgt_4,153,694). coords(lbgt_4,122,183,694).  visible(lbgt_4,694). happensAt(walking(lbgt_4),694).
orientation(lbgt_3,61,695). coords(lbgt_3,146,162,695).  visible(lbgt_3,695). happensAt(inactive(lbgt_3),695). orientation(lbgt_4,152,695). coords(lbgt_4,121,182,695).  visible(lbgt_4,695). happensAt(walking(lbgt_4),695).
orientation(lbgt_3,61,696). coords(lbgt_3,146,162,696).  visible(lbgt_3,696). happensAt(inactive(lbgt_3),696). orientation(lbgt_4,152,696). coords(lbgt_4,121,182,696).  visible(lbgt_4,696). happensAt(walking(lbgt_4),696).
orientation(lbgt_3,61,697). coords(lbgt_3,146,162,697).  visible(lbgt_3,697). happensAt(inactive(lbgt_3),697). orientation(lbgt_4,152,697). coords(lbgt_4,121,181,697).  visible(lbgt_4,697). happensAt(walking(lbgt_4),697).
orientation(lbgt_3,61,698). coords(lbgt_3,146,162,698).  visible(lbgt_3,698). happensAt(inactive(lbgt_3),698). orientation(lbgt_4,154,698). coords(lbgt_4,121,180,698).  visible(lbgt_4,698). happensAt(walking(lbgt_4),698).
orientation(lbgt_3,61,699). coords(lbgt_3,146,162,699).  visible(lbgt_3,699). happensAt(inactive(lbgt_3),699). orientation(lbgt_4,152,699). coords(lbgt_4,119,179,699).  visible(lbgt_4,699). happensAt(walking(lbgt_4),699).
orientation(lbgt_3,61,700). coords(lbgt_3,146,162,700).  visible(lbgt_3,700). happensAt(inactive(lbgt_3),700). orientation(lbgt_4,152,700). coords(lbgt_4,118,177,700).  visible(lbgt_4,700). happensAt(walking(lbgt_4),700).
orientation(lbgt_3,61,701). coords(lbgt_3,146,162,701).  visible(lbgt_3,701). happensAt(inactive(lbgt_3),701). orientation(lbgt_4,157,701). coords(lbgt_4,117,175,701).  visible(lbgt_4,701). happensAt(walking(lbgt_4),701).
orientation(lbgt_3,61,702). coords(lbgt_3,146,162,702).  visible(lbgt_3,702). happensAt(inactive(lbgt_3),702). orientation(lbgt_4,157,702). coords(lbgt_4,116,173,702).  visible(lbgt_4,702). happensAt(walking(lbgt_4),702).
orientation(lbgt_3,61,703). coords(lbgt_3,146,162,703).  visible(lbgt_3,703). happensAt(inactive(lbgt_3),703). orientation(lbgt_4,157,703). coords(lbgt_4,116,172,703).  visible(lbgt_4,703). happensAt(walking(lbgt_4),703).
#show holdsAt/2.