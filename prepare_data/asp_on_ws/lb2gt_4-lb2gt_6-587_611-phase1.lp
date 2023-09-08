
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

	orientation(lb2gt_4,130,587). coords(lb2gt_4,142,36,587).  visible(lb2gt_4,587). happensAt(walking(lb2gt_4),587). orientation(lb2gt_6,151,587). coords(lb2gt_6,72,122,587).  visible(lb2gt_6,587). happensAt(inactive(lb2gt_6),587).
orientation(lb2gt_4,130,588). coords(lb2gt_4,143,36,588).  visible(lb2gt_4,588). happensAt(walking(lb2gt_4),588). orientation(lb2gt_6,151,588). coords(lb2gt_6,72,122,588).  visible(lb2gt_6,588). happensAt(inactive(lb2gt_6),588).
orientation(lb2gt_4,130,589). coords(lb2gt_4,143,36,589).  visible(lb2gt_4,589). happensAt(walking(lb2gt_4),589). orientation(lb2gt_6,151,589). coords(lb2gt_6,72,122,589).  visible(lb2gt_6,589). happensAt(inactive(lb2gt_6),589).
orientation(lb2gt_4,130,590). coords(lb2gt_4,143,35,590).  visible(lb2gt_4,590). happensAt(walking(lb2gt_4),590). orientation(lb2gt_6,151,590). coords(lb2gt_6,72,122,590).  visible(lb2gt_6,590). happensAt(inactive(lb2gt_6),590).
orientation(lb2gt_4,130,591). coords(lb2gt_4,145,35,591).  visible(lb2gt_4,591). happensAt(walking(lb2gt_4),591). orientation(lb2gt_6,151,591). coords(lb2gt_6,72,122,591).  visible(lb2gt_6,591). happensAt(inactive(lb2gt_6),591).
orientation(lb2gt_4,130,592). coords(lb2gt_4,145,35,592).  visible(lb2gt_4,592). happensAt(walking(lb2gt_4),592). orientation(lb2gt_6,151,592). coords(lb2gt_6,72,122,592).  visible(lb2gt_6,592). happensAt(inactive(lb2gt_6),592).
orientation(lb2gt_4,130,593). coords(lb2gt_4,145,34,593).  visible(lb2gt_4,593). happensAt(walking(lb2gt_4),593). orientation(lb2gt_6,151,593). coords(lb2gt_6,72,122,593).  visible(lb2gt_6,593). happensAt(inactive(lb2gt_6),593).
orientation(lb2gt_4,130,594). coords(lb2gt_4,146,33,594).  visible(lb2gt_4,594). happensAt(walking(lb2gt_4),594). orientation(lb2gt_6,151,594). coords(lb2gt_6,72,122,594).  visible(lb2gt_6,594). happensAt(inactive(lb2gt_6),594).
orientation(lb2gt_4,130,595). coords(lb2gt_4,146,33,595).  visible(lb2gt_4,595). happensAt(walking(lb2gt_4),595). orientation(lb2gt_6,151,595). coords(lb2gt_6,72,122,595).  visible(lb2gt_6,595). happensAt(inactive(lb2gt_6),595).
orientation(lb2gt_4,130,596). coords(lb2gt_4,147,32,596).  visible(lb2gt_4,596). happensAt(walking(lb2gt_4),596). orientation(lb2gt_6,151,596). coords(lb2gt_6,72,122,596).  visible(lb2gt_6,596). happensAt(inactive(lb2gt_6),596).
orientation(lb2gt_4,130,597). coords(lb2gt_4,147,31,597).  visible(lb2gt_4,597). happensAt(walking(lb2gt_4),597). orientation(lb2gt_6,151,597). coords(lb2gt_6,72,122,597).  visible(lb2gt_6,597). happensAt(inactive(lb2gt_6),597).
orientation(lb2gt_4,130,598). coords(lb2gt_4,147,30,598).  visible(lb2gt_4,598). happensAt(walking(lb2gt_4),598). orientation(lb2gt_6,151,598). coords(lb2gt_6,72,122,598).  visible(lb2gt_6,598). happensAt(inactive(lb2gt_6),598).
orientation(lb2gt_4,130,599). coords(lb2gt_4,147,30,599).  visible(lb2gt_4,599). happensAt(walking(lb2gt_4),599). orientation(lb2gt_6,151,599). coords(lb2gt_6,72,122,599).  visible(lb2gt_6,599). happensAt(inactive(lb2gt_6),599).
orientation(lb2gt_4,130,600). coords(lb2gt_4,148,29,600).  visible(lb2gt_4,600). happensAt(walking(lb2gt_4),600). orientation(lb2gt_6,151,600). coords(lb2gt_6,72,122,600).  visible(lb2gt_6,600). happensAt(inactive(lb2gt_6),600).
orientation(lb2gt_4,130,601). coords(lb2gt_4,148,28,601).  visible(lb2gt_4,601). happensAt(walking(lb2gt_4),601). orientation(lb2gt_6,151,601). coords(lb2gt_6,72,122,601).  visible(lb2gt_6,601). happensAt(inactive(lb2gt_6),601).
orientation(lb2gt_4,130,602). coords(lb2gt_4,148,28,602).  visible(lb2gt_4,602). happensAt(walking(lb2gt_4),602). orientation(lb2gt_6,151,602). coords(lb2gt_6,72,122,602).  visible(lb2gt_6,602). happensAt(inactive(lb2gt_6),602).
orientation(lb2gt_4,130,603). coords(lb2gt_4,149,27,603).  visible(lb2gt_4,603). happensAt(walking(lb2gt_4),603). orientation(lb2gt_6,151,603). coords(lb2gt_6,72,122,603).  visible(lb2gt_6,603). happensAt(inactive(lb2gt_6),603).
orientation(lb2gt_4,130,604). coords(lb2gt_4,149,26,604).  visible(lb2gt_4,604). happensAt(walking(lb2gt_4),604). orientation(lb2gt_6,151,604). coords(lb2gt_6,72,122,604).  visible(lb2gt_6,604). happensAt(inactive(lb2gt_6),604).
orientation(lb2gt_4,130,605). coords(lb2gt_4,150,26,605).  visible(lb2gt_4,605). happensAt(walking(lb2gt_4),605). orientation(lb2gt_6,151,605). coords(lb2gt_6,72,122,605).  visible(lb2gt_6,605). happensAt(inactive(lb2gt_6),605).
orientation(lb2gt_4,130,606). coords(lb2gt_4,151,25,606).  visible(lb2gt_4,606). happensAt(walking(lb2gt_4),606). orientation(lb2gt_6,151,606). coords(lb2gt_6,72,122,606).  visible(lb2gt_6,606). happensAt(inactive(lb2gt_6),606).
orientation(lb2gt_4,130,607). coords(lb2gt_4,151,25,607).  visible(lb2gt_4,607). happensAt(walking(lb2gt_4),607). orientation(lb2gt_6,151,607). coords(lb2gt_6,72,122,607).  visible(lb2gt_6,607). happensAt(inactive(lb2gt_6),607).
orientation(lb2gt_4,130,608). coords(lb2gt_4,151,25,608).  visible(lb2gt_4,608). happensAt(walking(lb2gt_4),608). orientation(lb2gt_6,151,608). coords(lb2gt_6,72,122,608).  visible(lb2gt_6,608). happensAt(inactive(lb2gt_6),608).
orientation(lb2gt_4,130,609). coords(lb2gt_4,151,24,609).  visible(lb2gt_4,609). happensAt(walking(lb2gt_4),609). orientation(lb2gt_6,151,609). coords(lb2gt_6,72,122,609).  visible(lb2gt_6,609). happensAt(inactive(lb2gt_6),609).
orientation(lb2gt_4,130,610). coords(lb2gt_4,152,24,610).  visible(lb2gt_4,610). happensAt(walking(lb2gt_4),610). orientation(lb2gt_6,151,610). coords(lb2gt_6,72,122,610).  visible(lb2gt_6,610). happensAt(inactive(lb2gt_6),610).
orientation(lb2gt_4,130,611). coords(lb2gt_4,152,24,611).  visible(lb2gt_4,611). happensAt(walking(lb2gt_4),611). orientation(lb2gt_6,151,611). coords(lb2gt_6,72,122,611).  visible(lb2gt_6,611). happensAt(inactive(lb2gt_6),611).
#show holdsAt/2.