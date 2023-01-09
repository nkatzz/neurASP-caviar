
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

	orientation(wk2gt_1,125,583). coords(wk2gt_1,115,53,583).  visible(wk2gt_1,583). happensAt(inactive(wk2gt_1),583). orientation(wk2gt_4,152,583). coords(wk2gt_4,39,108,583).  visible(wk2gt_4,583). happensAt(walking(wk2gt_4),583).
orientation(wk2gt_1,125,584). coords(wk2gt_1,115,53,584).  visible(wk2gt_1,584). happensAt(inactive(wk2gt_1),584). orientation(wk2gt_4,152,584). coords(wk2gt_4,39,108,584).  visible(wk2gt_4,584). happensAt(walking(wk2gt_4),584).
orientation(wk2gt_1,125,585). coords(wk2gt_1,115,53,585).  visible(wk2gt_1,585). happensAt(inactive(wk2gt_1),585). orientation(wk2gt_4,152,585). coords(wk2gt_4,39,109,585).  visible(wk2gt_4,585). happensAt(walking(wk2gt_4),585).
orientation(wk2gt_1,125,586). coords(wk2gt_1,115,53,586).  visible(wk2gt_1,586). happensAt(inactive(wk2gt_1),586). orientation(wk2gt_4,152,586). coords(wk2gt_4,38,110,586).  visible(wk2gt_4,586). happensAt(walking(wk2gt_4),586).
orientation(wk2gt_1,125,587). coords(wk2gt_1,115,53,587).  visible(wk2gt_1,587). happensAt(inactive(wk2gt_1),587). orientation(wk2gt_4,152,587). coords(wk2gt_4,38,110,587).  visible(wk2gt_4,587). happensAt(walking(wk2gt_4),587).
orientation(wk2gt_1,125,588). coords(wk2gt_1,115,53,588).  visible(wk2gt_1,588). happensAt(inactive(wk2gt_1),588). orientation(wk2gt_4,152,588). coords(wk2gt_4,37,110,588).  visible(wk2gt_4,588). happensAt(walking(wk2gt_4),588).
orientation(wk2gt_1,125,589). coords(wk2gt_1,115,53,589).  visible(wk2gt_1,589). happensAt(inactive(wk2gt_1),589). orientation(wk2gt_4,152,589). coords(wk2gt_4,37,110,589).  visible(wk2gt_4,589). happensAt(walking(wk2gt_4),589).
orientation(wk2gt_1,125,590). coords(wk2gt_1,115,53,590).  visible(wk2gt_1,590). happensAt(inactive(wk2gt_1),590). orientation(wk2gt_4,152,590). coords(wk2gt_4,36,110,590).  visible(wk2gt_4,590). happensAt(walking(wk2gt_4),590).
orientation(wk2gt_1,125,591). coords(wk2gt_1,115,53,591).  visible(wk2gt_1,591). happensAt(inactive(wk2gt_1),591). orientation(wk2gt_4,152,591). coords(wk2gt_4,36,111,591).  visible(wk2gt_4,591). happensAt(walking(wk2gt_4),591).
orientation(wk2gt_1,125,592). coords(wk2gt_1,115,53,592).  visible(wk2gt_1,592). happensAt(inactive(wk2gt_1),592). orientation(wk2gt_4,152,592). coords(wk2gt_4,36,111,592).  visible(wk2gt_4,592). happensAt(walking(wk2gt_4),592).
orientation(wk2gt_1,125,593). coords(wk2gt_1,115,53,593).  visible(wk2gt_1,593). happensAt(inactive(wk2gt_1),593). orientation(wk2gt_4,152,593). coords(wk2gt_4,35,111,593).  visible(wk2gt_4,593). happensAt(walking(wk2gt_4),593).
orientation(wk2gt_1,125,594). coords(wk2gt_1,115,53,594).  visible(wk2gt_1,594). happensAt(inactive(wk2gt_1),594). orientation(wk2gt_4,152,594). coords(wk2gt_4,35,111,594).  visible(wk2gt_4,594). happensAt(walking(wk2gt_4),594).
orientation(wk2gt_1,125,595). coords(wk2gt_1,115,53,595).  visible(wk2gt_1,595). happensAt(inactive(wk2gt_1),595). orientation(wk2gt_4,152,595). coords(wk2gt_4,35,111,595).  visible(wk2gt_4,595). happensAt(walking(wk2gt_4),595).
orientation(wk2gt_1,125,596). coords(wk2gt_1,115,53,596).  visible(wk2gt_1,596). happensAt(inactive(wk2gt_1),596). orientation(wk2gt_4,152,596). coords(wk2gt_4,35,111,596).  visible(wk2gt_4,596). happensAt(walking(wk2gt_4),596).
orientation(wk2gt_1,125,597). coords(wk2gt_1,115,53,597).  visible(wk2gt_1,597). happensAt(inactive(wk2gt_1),597). orientation(wk2gt_4,152,597). coords(wk2gt_4,35,112,597).  visible(wk2gt_4,597). happensAt(walking(wk2gt_4),597).
orientation(wk2gt_1,125,598). coords(wk2gt_1,115,53,598).  visible(wk2gt_1,598). happensAt(inactive(wk2gt_1),598). orientation(wk2gt_4,152,598). coords(wk2gt_4,35,112,598).  visible(wk2gt_4,598). happensAt(walking(wk2gt_4),598).
orientation(wk2gt_1,125,599). coords(wk2gt_1,115,53,599).  visible(wk2gt_1,599). happensAt(inactive(wk2gt_1),599). orientation(wk2gt_4,152,599). coords(wk2gt_4,35,113,599).  visible(wk2gt_4,599). happensAt(walking(wk2gt_4),599).
orientation(wk2gt_1,125,600). coords(wk2gt_1,115,53,600).  visible(wk2gt_1,600). happensAt(inactive(wk2gt_1),600). orientation(wk2gt_4,152,600). coords(wk2gt_4,35,113,600).  visible(wk2gt_4,600). happensAt(walking(wk2gt_4),600).
orientation(wk2gt_1,125,601). coords(wk2gt_1,115,53,601).  visible(wk2gt_1,601). happensAt(inactive(wk2gt_1),601). orientation(wk2gt_4,157,601). coords(wk2gt_4,35,113,601).  visible(wk2gt_4,601). happensAt(walking(wk2gt_4),601).
orientation(wk2gt_1,125,602). coords(wk2gt_1,115,53,602).  visible(wk2gt_1,602). happensAt(inactive(wk2gt_1),602). orientation(wk2gt_4,157,602). coords(wk2gt_4,35,113,602).  visible(wk2gt_4,602). happensAt(walking(wk2gt_4),602).
orientation(wk2gt_1,125,603). coords(wk2gt_1,115,53,603).  visible(wk2gt_1,603). happensAt(inactive(wk2gt_1),603). orientation(wk2gt_4,161,603). coords(wk2gt_4,32,113,603).  visible(wk2gt_4,603). happensAt(walking(wk2gt_4),603).
orientation(wk2gt_1,125,604). coords(wk2gt_1,115,53,604).  visible(wk2gt_1,604). happensAt(inactive(wk2gt_1),604). orientation(wk2gt_4,161,604). coords(wk2gt_4,32,113,604).  visible(wk2gt_4,604). happensAt(walking(wk2gt_4),604).
orientation(wk2gt_1,125,605). coords(wk2gt_1,115,53,605).  visible(wk2gt_1,605). happensAt(inactive(wk2gt_1),605). orientation(wk2gt_4,161,605). coords(wk2gt_4,32,115,605).  visible(wk2gt_4,605). happensAt(walking(wk2gt_4),605).
orientation(wk2gt_1,125,606). coords(wk2gt_1,115,53,606).  visible(wk2gt_1,606). happensAt(inactive(wk2gt_1),606). orientation(wk2gt_4,161,606). coords(wk2gt_4,32,115,606).  visible(wk2gt_4,606). happensAt(walking(wk2gt_4),606).
orientation(wk2gt_1,125,607). coords(wk2gt_1,115,53,607).  visible(wk2gt_1,607). happensAt(inactive(wk2gt_1),607). orientation(wk2gt_4,161,607). coords(wk2gt_4,32,116,607).  visible(wk2gt_4,607). happensAt(walking(wk2gt_4),607).
#show holdsAt/2.