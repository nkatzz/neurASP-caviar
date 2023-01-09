
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

	orientation(mwt2gt_1,153,589). coords(mwt2gt_1,31,155,589).  visible(mwt2gt_1,589). happensAt(walking(mwt2gt_1),589). orientation(mwt2gt_2,152,589). coords(mwt2gt_2,26,165,589).  visible(mwt2gt_2,589). happensAt(walking(mwt2gt_2),589).
orientation(mwt2gt_1,153,590). coords(mwt2gt_1,31,155,590).  visible(mwt2gt_1,590). happensAt(walking(mwt2gt_1),590). orientation(mwt2gt_2,152,590). coords(mwt2gt_2,26,165,590).  visible(mwt2gt_2,590). happensAt(walking(mwt2gt_2),590).
orientation(mwt2gt_1,153,591). coords(mwt2gt_1,29,155,591).  visible(mwt2gt_1,591). happensAt(walking(mwt2gt_1),591). orientation(mwt2gt_2,152,591). coords(mwt2gt_2,26,165,591).  visible(mwt2gt_2,591). happensAt(walking(mwt2gt_2),591).
orientation(mwt2gt_1,153,592). coords(mwt2gt_1,29,155,592).  visible(mwt2gt_1,592). happensAt(walking(mwt2gt_1),592). orientation(mwt2gt_2,152,592). coords(mwt2gt_2,26,165,592).  visible(mwt2gt_2,592). happensAt(walking(mwt2gt_2),592).
orientation(mwt2gt_1,153,593). coords(mwt2gt_1,29,155,593).  visible(mwt2gt_1,593). happensAt(walking(mwt2gt_1),593). orientation(mwt2gt_2,152,593). coords(mwt2gt_2,26,165,593).  visible(mwt2gt_2,593). happensAt(walking(mwt2gt_2),593).
orientation(mwt2gt_1,153,594). coords(mwt2gt_1,29,155,594).  visible(mwt2gt_1,594). happensAt(walking(mwt2gt_1),594). orientation(mwt2gt_2,152,594). coords(mwt2gt_2,26,165,594).  visible(mwt2gt_2,594). happensAt(walking(mwt2gt_2),594).
orientation(mwt2gt_1,153,595). coords(mwt2gt_1,29,155,595).  visible(mwt2gt_1,595). happensAt(walking(mwt2gt_1),595). orientation(mwt2gt_2,152,595). coords(mwt2gt_2,26,165,595).  visible(mwt2gt_2,595). happensAt(walking(mwt2gt_2),595).
orientation(mwt2gt_1,153,596). coords(mwt2gt_1,29,155,596).  visible(mwt2gt_1,596). happensAt(walking(mwt2gt_1),596). orientation(mwt2gt_2,152,596). coords(mwt2gt_2,26,165,596).  visible(mwt2gt_2,596). happensAt(walking(mwt2gt_2),596).
orientation(mwt2gt_1,153,597). coords(mwt2gt_1,29,155,597).  visible(mwt2gt_1,597). happensAt(walking(mwt2gt_1),597). orientation(mwt2gt_2,152,597). coords(mwt2gt_2,26,165,597).  visible(mwt2gt_2,597). happensAt(walking(mwt2gt_2),597).
orientation(mwt2gt_1,153,598). coords(mwt2gt_1,28,155,598).  visible(mwt2gt_1,598). happensAt(walking(mwt2gt_1),598). orientation(mwt2gt_2,152,598). coords(mwt2gt_2,27,164,598).  visible(mwt2gt_2,598). happensAt(walking(mwt2gt_2),598).
orientation(mwt2gt_1,153,599). coords(mwt2gt_1,29,155,599).  visible(mwt2gt_1,599). happensAt(walking(mwt2gt_1),599). orientation(mwt2gt_2,152,599). coords(mwt2gt_2,27,164,599).  visible(mwt2gt_2,599). happensAt(walking(mwt2gt_2),599).
orientation(mwt2gt_1,153,600). coords(mwt2gt_1,28,156,600).  visible(mwt2gt_1,600). happensAt(walking(mwt2gt_1),600). orientation(mwt2gt_2,152,600). coords(mwt2gt_2,27,164,600).  visible(mwt2gt_2,600). happensAt(walking(mwt2gt_2),600).
orientation(mwt2gt_1,153,601). coords(mwt2gt_1,28,156,601).  visible(mwt2gt_1,601). happensAt(walking(mwt2gt_1),601). orientation(mwt2gt_2,152,601). coords(mwt2gt_2,26,164,601).  visible(mwt2gt_2,601). happensAt(walking(mwt2gt_2),601).
orientation(mwt2gt_1,153,602). coords(mwt2gt_1,28,156,602).  visible(mwt2gt_1,602). happensAt(walking(mwt2gt_1),602). orientation(mwt2gt_2,152,602). coords(mwt2gt_2,26,164,602).  visible(mwt2gt_2,602). happensAt(walking(mwt2gt_2),602).
orientation(mwt2gt_1,153,603). coords(mwt2gt_1,27,156,603).  visible(mwt2gt_1,603). happensAt(walking(mwt2gt_1),603). orientation(mwt2gt_2,152,603). coords(mwt2gt_2,26,164,603).  visible(mwt2gt_2,603). happensAt(walking(mwt2gt_2),603).
orientation(mwt2gt_1,153,604). coords(mwt2gt_1,29,156,604).  visible(mwt2gt_1,604). happensAt(walking(mwt2gt_1),604). orientation(mwt2gt_2,152,604). coords(mwt2gt_2,26,164,604).  visible(mwt2gt_2,604). happensAt(walking(mwt2gt_2),604).
orientation(mwt2gt_1,153,605). coords(mwt2gt_1,27,155,605).  visible(mwt2gt_1,605). happensAt(walking(mwt2gt_1),605). orientation(mwt2gt_2,152,605). coords(mwt2gt_2,26,164,605).  visible(mwt2gt_2,605). happensAt(walking(mwt2gt_2),605).
orientation(mwt2gt_1,153,606). coords(mwt2gt_1,26,155,606).  visible(mwt2gt_1,606). happensAt(walking(mwt2gt_1),606). orientation(mwt2gt_2,152,606). coords(mwt2gt_2,26,164,606).  visible(mwt2gt_2,606). happensAt(walking(mwt2gt_2),606).
orientation(mwt2gt_1,153,607). coords(mwt2gt_1,26,155,607).  visible(mwt2gt_1,607). happensAt(walking(mwt2gt_1),607). orientation(mwt2gt_2,152,607). coords(mwt2gt_2,26,164,607).  visible(mwt2gt_2,607). happensAt(walking(mwt2gt_2),607).
orientation(mwt2gt_1,153,608). coords(mwt2gt_1,27,155,608).  visible(mwt2gt_1,608). happensAt(walking(mwt2gt_1),608). orientation(mwt2gt_2,152,608). coords(mwt2gt_2,26,164,608).  visible(mwt2gt_2,608). happensAt(walking(mwt2gt_2),608).
orientation(mwt2gt_1,153,609). coords(mwt2gt_1,27,155,609).  visible(mwt2gt_1,609). happensAt(walking(mwt2gt_1),609). orientation(mwt2gt_2,152,609). coords(mwt2gt_2,26,164,609).  visible(mwt2gt_2,609). happensAt(walking(mwt2gt_2),609).
orientation(mwt2gt_1,153,610). coords(mwt2gt_1,26,155,610).  visible(mwt2gt_1,610). happensAt(walking(mwt2gt_1),610). orientation(mwt2gt_2,152,610). coords(mwt2gt_2,26,164,610).  visible(mwt2gt_2,610). happensAt(walking(mwt2gt_2),610).
orientation(mwt2gt_1,153,611). coords(mwt2gt_1,27,155,611).  visible(mwt2gt_1,611). happensAt(walking(mwt2gt_1),611). orientation(mwt2gt_2,152,611). coords(mwt2gt_2,26,164,611).  visible(mwt2gt_2,611). happensAt(walking(mwt2gt_2),611).
orientation(mwt2gt_1,153,612). coords(mwt2gt_1,27,155,612).  visible(mwt2gt_1,612). happensAt(walking(mwt2gt_1),612). orientation(mwt2gt_2,152,612). coords(mwt2gt_2,26,162,612).  visible(mwt2gt_2,612). happensAt(walking(mwt2gt_2),612).
orientation(mwt2gt_1,153,613). coords(mwt2gt_1,27,155,613).  visible(mwt2gt_1,613). happensAt(walking(mwt2gt_1),613). orientation(mwt2gt_2,152,613). coords(mwt2gt_2,26,162,613).  visible(mwt2gt_2,613). happensAt(walking(mwt2gt_2),613).
#show holdsAt/2.