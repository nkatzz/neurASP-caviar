
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

	orientation(ms3ggt_0,168,584). coords(ms3ggt_0,73,192,584).  visible(ms3ggt_0,584). happensAt(inactive(ms3ggt_0),584). orientation(ms3ggt_2,150,584). coords(ms3ggt_2,83,83,584).  visible(ms3ggt_2,584). happensAt(walking(ms3ggt_2),584).
orientation(ms3ggt_0,173,585). coords(ms3ggt_0,75,192,585).  visible(ms3ggt_0,585). happensAt(inactive(ms3ggt_0),585). orientation(ms3ggt_2,150,585). coords(ms3ggt_2,83,83,585).  visible(ms3ggt_2,585). happensAt(walking(ms3ggt_2),585).
orientation(ms3ggt_0,173,586). coords(ms3ggt_0,75,192,586).  visible(ms3ggt_0,586). happensAt(inactive(ms3ggt_0),586). orientation(ms3ggt_2,150,586). coords(ms3ggt_2,83,83,586).  visible(ms3ggt_2,586). happensAt(walking(ms3ggt_2),586).
orientation(ms3ggt_0,173,587). coords(ms3ggt_0,77,192,587).  visible(ms3ggt_0,587). happensAt(walking(ms3ggt_0),587). orientation(ms3ggt_2,150,587). coords(ms3ggt_2,83,83,587).  visible(ms3ggt_2,587). happensAt(walking(ms3ggt_2),587).
orientation(ms3ggt_0,173,588). coords(ms3ggt_0,77,192,588).  visible(ms3ggt_0,588). happensAt(walking(ms3ggt_0),588). orientation(ms3ggt_2,150,588). coords(ms3ggt_2,83,81,588).  visible(ms3ggt_2,588). happensAt(walking(ms3ggt_2),588).
orientation(ms3ggt_0,173,589). coords(ms3ggt_0,77,192,589).  visible(ms3ggt_0,589). happensAt(walking(ms3ggt_0),589). orientation(ms3ggt_2,150,589). coords(ms3ggt_2,83,81,589).  visible(ms3ggt_2,589). happensAt(walking(ms3ggt_2),589).
orientation(ms3ggt_0,173,590). coords(ms3ggt_0,77,192,590).  visible(ms3ggt_0,590). happensAt(walking(ms3ggt_0),590). orientation(ms3ggt_2,150,590). coords(ms3ggt_2,83,80,590).  visible(ms3ggt_2,590). happensAt(walking(ms3ggt_2),590).
orientation(ms3ggt_0,173,591). coords(ms3ggt_0,77,192,591).  visible(ms3ggt_0,591). happensAt(walking(ms3ggt_0),591). orientation(ms3ggt_2,150,591). coords(ms3ggt_2,83,80,591).  visible(ms3ggt_2,591). happensAt(walking(ms3ggt_2),591).
orientation(ms3ggt_0,173,592). coords(ms3ggt_0,77,192,592).  visible(ms3ggt_0,592). happensAt(walking(ms3ggt_0),592). orientation(ms3ggt_2,150,592). coords(ms3ggt_2,83,80,592).  visible(ms3ggt_2,592). happensAt(walking(ms3ggt_2),592).
orientation(ms3ggt_0,173,593). coords(ms3ggt_0,77,192,593).  visible(ms3ggt_0,593). happensAt(walking(ms3ggt_0),593). orientation(ms3ggt_2,150,593). coords(ms3ggt_2,83,80,593).  visible(ms3ggt_2,593). happensAt(walking(ms3ggt_2),593).
orientation(ms3ggt_0,164,594). coords(ms3ggt_0,78,192,594).  visible(ms3ggt_0,594). happensAt(walking(ms3ggt_0),594). orientation(ms3ggt_2,150,594). coords(ms3ggt_2,83,79,594).  visible(ms3ggt_2,594). happensAt(walking(ms3ggt_2),594).
orientation(ms3ggt_0,164,595). coords(ms3ggt_0,79,192,595).  visible(ms3ggt_0,595). happensAt(walking(ms3ggt_0),595). orientation(ms3ggt_2,150,595). coords(ms3ggt_2,83,79,595).  visible(ms3ggt_2,595). happensAt(walking(ms3ggt_2),595).
orientation(ms3ggt_0,164,596). coords(ms3ggt_0,79,192,596).  happensAt(appear(ms3ggt_0),596). happensAt(walking(ms3ggt_0),596). orientation(ms3ggt_2,150,596). coords(ms3ggt_2,83,79,596).  visible(ms3ggt_2,596). happensAt(walking(ms3ggt_2),596).
orientation(ms3ggt_0,164,597). coords(ms3ggt_0,78,192,597).  visible(ms3ggt_0,597). happensAt(walking(ms3ggt_0),597). orientation(ms3ggt_2,150,597). coords(ms3ggt_2,83,79,597).  visible(ms3ggt_2,597). happensAt(walking(ms3ggt_2),597).
orientation(ms3ggt_0,164,598). coords(ms3ggt_0,79,192,598).  visible(ms3ggt_0,598). happensAt(walking(ms3ggt_0),598). orientation(ms3ggt_2,150,598). coords(ms3ggt_2,83,79,598).  visible(ms3ggt_2,598). happensAt(walking(ms3ggt_2),598).
orientation(ms3ggt_0,164,599). coords(ms3ggt_0,79,192,599).  visible(ms3ggt_0,599). happensAt(walking(ms3ggt_0),599). orientation(ms3ggt_2,142,599). coords(ms3ggt_2,83,79,599).  visible(ms3ggt_2,599). happensAt(walking(ms3ggt_2),599).
orientation(ms3ggt_0,164,600). coords(ms3ggt_0,79,192,600).  visible(ms3ggt_0,600). happensAt(walking(ms3ggt_0),600). orientation(ms3ggt_2,142,600). coords(ms3ggt_2,83,78,600).  visible(ms3ggt_2,600). happensAt(walking(ms3ggt_2),600).
orientation(ms3ggt_0,164,601). coords(ms3ggt_0,80,192,601).  visible(ms3ggt_0,601). happensAt(walking(ms3ggt_0),601). orientation(ms3ggt_2,142,601). coords(ms3ggt_2,83,78,601).  visible(ms3ggt_2,601). happensAt(walking(ms3ggt_2),601).
orientation(ms3ggt_0,164,602). coords(ms3ggt_0,80,191,602).  visible(ms3ggt_0,602). happensAt(walking(ms3ggt_0),602). orientation(ms3ggt_2,142,602). coords(ms3ggt_2,83,77,602).  visible(ms3ggt_2,602). happensAt(walking(ms3ggt_2),602).
orientation(ms3ggt_0,164,603). coords(ms3ggt_0,83,191,603).  visible(ms3ggt_0,603). happensAt(walking(ms3ggt_0),603). orientation(ms3ggt_2,142,603). coords(ms3ggt_2,83,76,603).  visible(ms3ggt_2,603). happensAt(walking(ms3ggt_2),603).
orientation(ms3ggt_0,160,604). coords(ms3ggt_0,83,191,604).  visible(ms3ggt_0,604). happensAt(walking(ms3ggt_0),604). orientation(ms3ggt_2,142,604). coords(ms3ggt_2,83,76,604).  visible(ms3ggt_2,604). happensAt(walking(ms3ggt_2),604).
orientation(ms3ggt_0,160,605). coords(ms3ggt_0,85,191,605).  visible(ms3ggt_0,605). happensAt(walking(ms3ggt_0),605). orientation(ms3ggt_2,142,605). coords(ms3ggt_2,83,75,605).  visible(ms3ggt_2,605). happensAt(walking(ms3ggt_2),605).
orientation(ms3ggt_0,165,606). coords(ms3ggt_0,86,191,606).  visible(ms3ggt_0,606). happensAt(walking(ms3ggt_0),606). orientation(ms3ggt_2,142,606). coords(ms3ggt_2,83,72,606).  visible(ms3ggt_2,606). happensAt(walking(ms3ggt_2),606).
orientation(ms3ggt_0,165,607). coords(ms3ggt_0,88,191,607).  visible(ms3ggt_0,607). happensAt(walking(ms3ggt_0),607). orientation(ms3ggt_2,142,607). coords(ms3ggt_2,83,71,607).  visible(ms3ggt_2,607). happensAt(walking(ms3ggt_2),607).
orientation(ms3ggt_0,165,608). coords(ms3ggt_0,88,191,608).  visible(ms3ggt_0,608). happensAt(walking(ms3ggt_0),608). orientation(ms3ggt_2,142,608). coords(ms3ggt_2,83,70,608).  visible(ms3ggt_2,608). happensAt(walking(ms3ggt_2),608).
#show holdsAt/2.