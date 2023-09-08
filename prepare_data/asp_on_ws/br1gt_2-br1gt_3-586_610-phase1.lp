
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

	orientation(br1gt_2,12,586). coords(br1gt_2,86,209,586).  visible(br1gt_2,586). happensAt(inactive(br1gt_2),586). orientation(br1gt_3,139,586). coords(br1gt_3,253,99,586).  visible(br1gt_3,586). happensAt(walking(br1gt_3),586).
orientation(br1gt_2,12,587). coords(br1gt_2,86,209,587).  visible(br1gt_2,587). happensAt(inactive(br1gt_2),587). orientation(br1gt_3,134,587). coords(br1gt_3,254,99,587).  visible(br1gt_3,587). happensAt(walking(br1gt_3),587).
orientation(br1gt_2,12,588). coords(br1gt_2,86,209,588).  visible(br1gt_2,588). happensAt(inactive(br1gt_2),588). orientation(br1gt_3,134,588). coords(br1gt_3,255,98,588).  visible(br1gt_3,588). happensAt(walking(br1gt_3),588).
orientation(br1gt_2,12,589). coords(br1gt_2,86,209,589).  visible(br1gt_2,589). happensAt(inactive(br1gt_2),589). orientation(br1gt_3,134,589). coords(br1gt_3,256,99,589).  visible(br1gt_3,589). happensAt(walking(br1gt_3),589).
orientation(br1gt_2,12,590). coords(br1gt_2,86,209,590).  visible(br1gt_2,590). happensAt(inactive(br1gt_2),590). orientation(br1gt_3,134,590). coords(br1gt_3,257,98,590).  visible(br1gt_3,590). happensAt(walking(br1gt_3),590).
orientation(br1gt_2,12,591). coords(br1gt_2,86,209,591).  visible(br1gt_2,591). happensAt(inactive(br1gt_2),591). orientation(br1gt_3,126,591). coords(br1gt_3,260,98,591).  visible(br1gt_3,591). happensAt(walking(br1gt_3),591).
orientation(br1gt_2,12,592). coords(br1gt_2,86,209,592).  visible(br1gt_2,592). happensAt(inactive(br1gt_2),592). orientation(br1gt_3,126,592). coords(br1gt_3,264,98,592).  visible(br1gt_3,592). happensAt(walking(br1gt_3),592).
orientation(br1gt_2,12,593). coords(br1gt_2,86,209,593).  visible(br1gt_2,593). happensAt(inactive(br1gt_2),593). orientation(br1gt_3,129,593). coords(br1gt_3,267,98,593).  visible(br1gt_3,593). happensAt(walking(br1gt_3),593).
orientation(br1gt_2,12,594). coords(br1gt_2,86,209,594).  visible(br1gt_2,594). happensAt(inactive(br1gt_2),594). orientation(br1gt_3,129,594). coords(br1gt_3,270,98,594).  visible(br1gt_3,594). happensAt(walking(br1gt_3),594).
orientation(br1gt_2,12,595). coords(br1gt_2,86,209,595).  visible(br1gt_2,595). happensAt(inactive(br1gt_2),595). orientation(br1gt_3,129,595). coords(br1gt_3,272,98,595).  visible(br1gt_3,595). happensAt(walking(br1gt_3),595).
orientation(br1gt_2,12,596). coords(br1gt_2,86,209,596).  visible(br1gt_2,596). happensAt(inactive(br1gt_2),596). orientation(br1gt_3,129,596). coords(br1gt_3,274,97,596).  visible(br1gt_3,596). happensAt(walking(br1gt_3),596).
orientation(br1gt_2,12,597). coords(br1gt_2,86,209,597).  visible(br1gt_2,597). happensAt(inactive(br1gt_2),597). orientation(br1gt_3,129,597). coords(br1gt_3,275,97,597).  visible(br1gt_3,597). happensAt(walking(br1gt_3),597).
orientation(br1gt_2,12,598). coords(br1gt_2,86,209,598).  visible(br1gt_2,598). happensAt(inactive(br1gt_2),598). orientation(br1gt_3,129,598). coords(br1gt_3,276,97,598).  visible(br1gt_3,598). happensAt(walking(br1gt_3),598).
orientation(br1gt_2,12,599). coords(br1gt_2,86,209,599).  visible(br1gt_2,599). happensAt(inactive(br1gt_2),599). orientation(br1gt_3,129,599). coords(br1gt_3,277,96,599).  visible(br1gt_3,599). happensAt(walking(br1gt_3),599).
orientation(br1gt_2,12,600). coords(br1gt_2,86,209,600).  visible(br1gt_2,600). happensAt(inactive(br1gt_2),600). orientation(br1gt_3,120,600). coords(br1gt_3,277,96,600).  visible(br1gt_3,600). happensAt(walking(br1gt_3),600).
orientation(br1gt_2,12,601). coords(br1gt_2,86,209,601).  visible(br1gt_2,601). happensAt(inactive(br1gt_2),601). orientation(br1gt_3,120,601). coords(br1gt_3,279,96,601).  visible(br1gt_3,601). happensAt(walking(br1gt_3),601).
orientation(br1gt_2,12,602). coords(br1gt_2,86,209,602).  visible(br1gt_2,602). happensAt(inactive(br1gt_2),602). orientation(br1gt_3,120,602). coords(br1gt_3,279,96,602).  visible(br1gt_3,602). happensAt(walking(br1gt_3),602).
orientation(br1gt_2,12,603). coords(br1gt_2,86,209,603).  visible(br1gt_2,603). happensAt(inactive(br1gt_2),603). orientation(br1gt_3,120,603). coords(br1gt_3,280,95,603).  visible(br1gt_3,603). happensAt(walking(br1gt_3),603).
orientation(br1gt_2,12,604). coords(br1gt_2,86,209,604).  visible(br1gt_2,604). happensAt(inactive(br1gt_2),604). orientation(br1gt_3,120,604). coords(br1gt_3,280,95,604).  visible(br1gt_3,604). happensAt(walking(br1gt_3),604).
orientation(br1gt_2,12,605). coords(br1gt_2,86,209,605).  visible(br1gt_2,605). happensAt(inactive(br1gt_2),605). orientation(br1gt_3,113,605). coords(br1gt_3,281,95,605).  visible(br1gt_3,605). happensAt(walking(br1gt_3),605).
orientation(br1gt_2,12,606). coords(br1gt_2,86,209,606).  visible(br1gt_2,606). happensAt(inactive(br1gt_2),606). orientation(br1gt_3,113,606). coords(br1gt_3,283,95,606).  visible(br1gt_3,606). happensAt(walking(br1gt_3),606).
orientation(br1gt_2,12,607). coords(br1gt_2,86,209,607).  visible(br1gt_2,607). happensAt(inactive(br1gt_2),607). orientation(br1gt_3,113,607). coords(br1gt_3,286,95,607).  visible(br1gt_3,607). happensAt(walking(br1gt_3),607).
orientation(br1gt_2,12,608). coords(br1gt_2,86,209,608).  visible(br1gt_2,608). happensAt(inactive(br1gt_2),608). orientation(br1gt_3,113,608). coords(br1gt_3,289,95,608).  visible(br1gt_3,608). happensAt(walking(br1gt_3),608).
orientation(br1gt_2,12,609). coords(br1gt_2,86,209,609).  visible(br1gt_2,609). happensAt(inactive(br1gt_2),609). orientation(br1gt_3,113,609). coords(br1gt_3,290,95,609).  visible(br1gt_3,609). happensAt(walking(br1gt_3),609).
orientation(br1gt_2,12,610). coords(br1gt_2,86,209,610).  visible(br1gt_2,610). happensAt(inactive(br1gt_2),610). orientation(br1gt_3,113,610). coords(br1gt_3,290,95,610).  visible(br1gt_3,610). happensAt(walking(br1gt_3),610).
#show holdsAt/2.