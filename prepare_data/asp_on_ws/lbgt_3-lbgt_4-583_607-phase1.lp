
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

	orientation(lbgt_3,61,583). coords(lbgt_3,146,162,583).  visible(lbgt_3,583). happensAt(inactive(lbgt_3),583). orientation(lbgt_4,166,583). coords(lbgt_4,120,152,583).  visible(lbgt_4,583). happensAt(active(lbgt_4),583).
orientation(lbgt_3,61,584). coords(lbgt_3,146,162,584).  visible(lbgt_3,584). happensAt(inactive(lbgt_3),584). orientation(lbgt_4,164,584). coords(lbgt_4,119,152,584).  visible(lbgt_4,584). happensAt(active(lbgt_4),584).
orientation(lbgt_3,61,585). coords(lbgt_3,146,162,585).  visible(lbgt_3,585). happensAt(inactive(lbgt_3),585). orientation(lbgt_4,161,585). coords(lbgt_4,119,152,585).  visible(lbgt_4,585). happensAt(active(lbgt_4),585).
orientation(lbgt_3,61,586). coords(lbgt_3,146,162,586).  visible(lbgt_3,586). happensAt(inactive(lbgt_3),586). orientation(lbgt_4,161,586). coords(lbgt_4,118,152,586).  visible(lbgt_4,586). happensAt(active(lbgt_4),586).
orientation(lbgt_3,61,587). coords(lbgt_3,146,162,587).  visible(lbgt_3,587). happensAt(inactive(lbgt_3),587). orientation(lbgt_4,161,587). coords(lbgt_4,118,152,587).  visible(lbgt_4,587). happensAt(active(lbgt_4),587).
orientation(lbgt_3,61,588). coords(lbgt_3,146,162,588).  visible(lbgt_3,588). happensAt(inactive(lbgt_3),588). orientation(lbgt_4,161,588). coords(lbgt_4,118,152,588).  visible(lbgt_4,588). happensAt(active(lbgt_4),588).
orientation(lbgt_3,61,589). coords(lbgt_3,146,162,589).  visible(lbgt_3,589). happensAt(inactive(lbgt_3),589). orientation(lbgt_4,161,589). coords(lbgt_4,118,152,589).  visible(lbgt_4,589). happensAt(active(lbgt_4),589).
orientation(lbgt_3,61,590). coords(lbgt_3,146,162,590).  visible(lbgt_3,590). happensAt(inactive(lbgt_3),590). orientation(lbgt_4,161,590). coords(lbgt_4,117,152,590).  visible(lbgt_4,590). happensAt(active(lbgt_4),590).
orientation(lbgt_3,61,591). coords(lbgt_3,146,162,591).  visible(lbgt_3,591). happensAt(inactive(lbgt_3),591). orientation(lbgt_4,161,591). coords(lbgt_4,117,152,591).  visible(lbgt_4,591). happensAt(active(lbgt_4),591).
orientation(lbgt_3,61,592). coords(lbgt_3,146,162,592).  visible(lbgt_3,592). happensAt(inactive(lbgt_3),592). orientation(lbgt_4,161,592). coords(lbgt_4,117,153,592).  visible(lbgt_4,592). happensAt(active(lbgt_4),592).
orientation(lbgt_3,61,593). coords(lbgt_3,146,162,593).  visible(lbgt_3,593). happensAt(inactive(lbgt_3),593). orientation(lbgt_4,161,593). coords(lbgt_4,117,153,593).  visible(lbgt_4,593). happensAt(active(lbgt_4),593).
orientation(lbgt_3,61,594). coords(lbgt_3,146,162,594).  visible(lbgt_3,594). happensAt(inactive(lbgt_3),594). orientation(lbgt_4,161,594). coords(lbgt_4,117,153,594).  visible(lbgt_4,594). happensAt(active(lbgt_4),594).
orientation(lbgt_3,61,595). coords(lbgt_3,146,162,595).  visible(lbgt_3,595). happensAt(inactive(lbgt_3),595). orientation(lbgt_4,161,595). coords(lbgt_4,117,153,595).  visible(lbgt_4,595). happensAt(active(lbgt_4),595).
orientation(lbgt_3,61,596). coords(lbgt_3,146,162,596).  visible(lbgt_3,596). happensAt(inactive(lbgt_3),596). orientation(lbgt_4,161,596). coords(lbgt_4,117,153,596).  visible(lbgt_4,596). happensAt(active(lbgt_4),596).
orientation(lbgt_3,61,597). coords(lbgt_3,146,162,597).  visible(lbgt_3,597). happensAt(inactive(lbgt_3),597). orientation(lbgt_4,161,597). coords(lbgt_4,117,153,597).  visible(lbgt_4,597). happensAt(active(lbgt_4),597).
orientation(lbgt_3,61,598). coords(lbgt_3,146,162,598).  visible(lbgt_3,598). happensAt(inactive(lbgt_3),598). orientation(lbgt_4,161,598). coords(lbgt_4,117,153,598).  visible(lbgt_4,598). happensAt(active(lbgt_4),598).
orientation(lbgt_3,61,599). coords(lbgt_3,146,162,599).  visible(lbgt_3,599). happensAt(inactive(lbgt_3),599). orientation(lbgt_4,161,599). coords(lbgt_4,117,153,599).  visible(lbgt_4,599). happensAt(active(lbgt_4),599).
orientation(lbgt_3,61,600). coords(lbgt_3,146,162,600).  visible(lbgt_3,600). happensAt(inactive(lbgt_3),600). orientation(lbgt_4,161,600). coords(lbgt_4,117,153,600).  visible(lbgt_4,600). happensAt(active(lbgt_4),600).
orientation(lbgt_3,61,601). coords(lbgt_3,146,162,601).  visible(lbgt_3,601). happensAt(inactive(lbgt_3),601). orientation(lbgt_4,161,601). coords(lbgt_4,117,153,601).  visible(lbgt_4,601). happensAt(active(lbgt_4),601).
orientation(lbgt_3,61,602). coords(lbgt_3,146,162,602).  visible(lbgt_3,602). happensAt(inactive(lbgt_3),602). orientation(lbgt_4,161,602). coords(lbgt_4,117,153,602).  visible(lbgt_4,602). happensAt(active(lbgt_4),602).
orientation(lbgt_3,61,603). coords(lbgt_3,146,162,603).  visible(lbgt_3,603). happensAt(inactive(lbgt_3),603). orientation(lbgt_4,161,603). coords(lbgt_4,117,153,603).  visible(lbgt_4,603). happensAt(active(lbgt_4),603).
orientation(lbgt_3,61,604). coords(lbgt_3,146,162,604).  visible(lbgt_3,604). happensAt(inactive(lbgt_3),604). orientation(lbgt_4,161,604). coords(lbgt_4,117,154,604).  visible(lbgt_4,604). happensAt(active(lbgt_4),604).
orientation(lbgt_3,61,605). coords(lbgt_3,146,162,605).  visible(lbgt_3,605). happensAt(inactive(lbgt_3),605). orientation(lbgt_4,161,605). coords(lbgt_4,116,154,605).  visible(lbgt_4,605). happensAt(active(lbgt_4),605).
orientation(lbgt_3,61,606). coords(lbgt_3,146,162,606).  visible(lbgt_3,606). happensAt(inactive(lbgt_3),606). orientation(lbgt_4,161,606). coords(lbgt_4,116,154,606).  visible(lbgt_4,606). happensAt(active(lbgt_4),606).
orientation(lbgt_3,61,607). coords(lbgt_3,146,162,607).  visible(lbgt_3,607). happensAt(inactive(lbgt_3),607). orientation(lbgt_4,161,607). coords(lbgt_4,116,154,607).  visible(lbgt_4,607). happensAt(active(lbgt_4),607).
#show holdsAt/2.