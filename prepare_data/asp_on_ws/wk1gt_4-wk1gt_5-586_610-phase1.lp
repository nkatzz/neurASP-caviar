
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

	orientation(wk1gt_4,157,586). coords(wk1gt_4,56,170,586).  visible(wk1gt_4,586). happensAt(inactive(wk1gt_4),586). orientation(wk1gt_5,162,586). coords(wk1gt_5,57,150,586).  visible(wk1gt_5,586). happensAt(inactive(wk1gt_5),586).
orientation(wk1gt_4,157,587). coords(wk1gt_4,56,170,587).  visible(wk1gt_4,587). happensAt(inactive(wk1gt_4),587). orientation(wk1gt_5,162,587). coords(wk1gt_5,57,150,587).  visible(wk1gt_5,587). happensAt(inactive(wk1gt_5),587).
orientation(wk1gt_4,157,588). coords(wk1gt_4,56,170,588).  visible(wk1gt_4,588). happensAt(inactive(wk1gt_4),588). orientation(wk1gt_5,162,588). coords(wk1gt_5,57,150,588).  visible(wk1gt_5,588). happensAt(inactive(wk1gt_5),588).
orientation(wk1gt_4,157,589). coords(wk1gt_4,56,170,589).  visible(wk1gt_4,589). happensAt(inactive(wk1gt_4),589). orientation(wk1gt_5,162,589). coords(wk1gt_5,57,150,589).  visible(wk1gt_5,589). happensAt(inactive(wk1gt_5),589).
orientation(wk1gt_4,157,590). coords(wk1gt_4,56,170,590).  visible(wk1gt_4,590). happensAt(inactive(wk1gt_4),590). orientation(wk1gt_5,162,590). coords(wk1gt_5,57,150,590).  visible(wk1gt_5,590). happensAt(inactive(wk1gt_5),590).
orientation(wk1gt_4,157,591). coords(wk1gt_4,56,170,591).  visible(wk1gt_4,591). happensAt(inactive(wk1gt_4),591). orientation(wk1gt_5,162,591). coords(wk1gt_5,57,150,591).  visible(wk1gt_5,591). happensAt(inactive(wk1gt_5),591).
orientation(wk1gt_4,157,592). coords(wk1gt_4,56,170,592).  visible(wk1gt_4,592). happensAt(inactive(wk1gt_4),592). orientation(wk1gt_5,162,592). coords(wk1gt_5,57,150,592).  visible(wk1gt_5,592). happensAt(inactive(wk1gt_5),592).
orientation(wk1gt_4,157,593). coords(wk1gt_4,56,170,593).  visible(wk1gt_4,593). happensAt(inactive(wk1gt_4),593). orientation(wk1gt_5,162,593). coords(wk1gt_5,57,150,593).  visible(wk1gt_5,593). happensAt(inactive(wk1gt_5),593).
orientation(wk1gt_4,157,594). coords(wk1gt_4,56,170,594).  visible(wk1gt_4,594). happensAt(inactive(wk1gt_4),594). orientation(wk1gt_5,162,594). coords(wk1gt_5,57,150,594).  visible(wk1gt_5,594). happensAt(inactive(wk1gt_5),594).
orientation(wk1gt_4,157,595). coords(wk1gt_4,56,170,595).  visible(wk1gt_4,595). happensAt(inactive(wk1gt_4),595). orientation(wk1gt_5,162,595). coords(wk1gt_5,57,150,595).  visible(wk1gt_5,595). happensAt(inactive(wk1gt_5),595).
orientation(wk1gt_4,157,596). coords(wk1gt_4,56,170,596).  visible(wk1gt_4,596). happensAt(inactive(wk1gt_4),596). orientation(wk1gt_5,162,596). coords(wk1gt_5,57,150,596).  visible(wk1gt_5,596). happensAt(inactive(wk1gt_5),596).
orientation(wk1gt_4,157,597). coords(wk1gt_4,56,170,597).  visible(wk1gt_4,597). happensAt(inactive(wk1gt_4),597). orientation(wk1gt_5,162,597). coords(wk1gt_5,57,150,597).  visible(wk1gt_5,597). happensAt(inactive(wk1gt_5),597).
orientation(wk1gt_4,157,598). coords(wk1gt_4,56,170,598).  visible(wk1gt_4,598). happensAt(inactive(wk1gt_4),598). orientation(wk1gt_5,162,598). coords(wk1gt_5,57,150,598).  visible(wk1gt_5,598). happensAt(inactive(wk1gt_5),598).
orientation(wk1gt_4,157,599). coords(wk1gt_4,56,170,599).  visible(wk1gt_4,599). happensAt(inactive(wk1gt_4),599). orientation(wk1gt_5,162,599). coords(wk1gt_5,57,150,599).  visible(wk1gt_5,599). happensAt(inactive(wk1gt_5),599).
orientation(wk1gt_4,157,600). coords(wk1gt_4,56,170,600).  visible(wk1gt_4,600). happensAt(inactive(wk1gt_4),600). orientation(wk1gt_5,162,600). coords(wk1gt_5,57,150,600).  visible(wk1gt_5,600). happensAt(inactive(wk1gt_5),600).
orientation(wk1gt_4,157,601). coords(wk1gt_4,56,170,601).  visible(wk1gt_4,601). happensAt(inactive(wk1gt_4),601). orientation(wk1gt_5,162,601). coords(wk1gt_5,57,150,601).  visible(wk1gt_5,601). happensAt(inactive(wk1gt_5),601).
orientation(wk1gt_4,157,602). coords(wk1gt_4,56,170,602).  visible(wk1gt_4,602). happensAt(inactive(wk1gt_4),602). orientation(wk1gt_5,162,602). coords(wk1gt_5,57,150,602).  visible(wk1gt_5,602). happensAt(inactive(wk1gt_5),602).
orientation(wk1gt_4,157,603). coords(wk1gt_4,56,170,603).  visible(wk1gt_4,603). happensAt(inactive(wk1gt_4),603). orientation(wk1gt_5,162,603). coords(wk1gt_5,57,150,603).  visible(wk1gt_5,603). happensAt(inactive(wk1gt_5),603).
orientation(wk1gt_4,157,604). coords(wk1gt_4,56,170,604).  visible(wk1gt_4,604). happensAt(inactive(wk1gt_4),604). orientation(wk1gt_5,162,604). coords(wk1gt_5,57,150,604).  visible(wk1gt_5,604). happensAt(inactive(wk1gt_5),604).
orientation(wk1gt_4,157,605). coords(wk1gt_4,56,170,605).  visible(wk1gt_4,605). happensAt(inactive(wk1gt_4),605). orientation(wk1gt_5,162,605). coords(wk1gt_5,57,150,605).  visible(wk1gt_5,605). happensAt(inactive(wk1gt_5),605).
orientation(wk1gt_4,157,606). coords(wk1gt_4,56,170,606).  visible(wk1gt_4,606). happensAt(inactive(wk1gt_4),606). orientation(wk1gt_5,162,606). coords(wk1gt_5,57,150,606).  visible(wk1gt_5,606). happensAt(inactive(wk1gt_5),606).
orientation(wk1gt_4,157,607). coords(wk1gt_4,56,170,607).  visible(wk1gt_4,607). happensAt(inactive(wk1gt_4),607). orientation(wk1gt_5,162,607). coords(wk1gt_5,57,150,607).  visible(wk1gt_5,607). happensAt(inactive(wk1gt_5),607).
orientation(wk1gt_4,157,608). coords(wk1gt_4,56,170,608).  visible(wk1gt_4,608). happensAt(inactive(wk1gt_4),608). orientation(wk1gt_5,162,608). coords(wk1gt_5,57,150,608).  visible(wk1gt_5,608). happensAt(inactive(wk1gt_5),608).
orientation(wk1gt_4,157,609). coords(wk1gt_4,56,170,609).  visible(wk1gt_4,609). happensAt(inactive(wk1gt_4),609). orientation(wk1gt_5,162,609). coords(wk1gt_5,57,150,609).  visible(wk1gt_5,609). happensAt(inactive(wk1gt_5),609).
orientation(wk1gt_4,157,610). coords(wk1gt_4,56,170,610).  happensAt(disappear(wk1gt_4),610). happensAt(inactive(wk1gt_4),610). orientation(wk1gt_5,162,610). coords(wk1gt_5,57,150,610).  happensAt(disappear(wk1gt_5),610). happensAt(inactive(wk1gt_5),610).
#show holdsAt/2.