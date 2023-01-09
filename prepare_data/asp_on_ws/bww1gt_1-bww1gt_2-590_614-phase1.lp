
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

	orientation(bww1gt_1,173,590). coords(bww1gt_1,100,249,590).  visible(bww1gt_1,590). happensAt(inactive(bww1gt_1),590). orientation(bww1gt_2,176,590). coords(bww1gt_2,135,211,590).  visible(bww1gt_2,590). happensAt(walking(bww1gt_2),590).
orientation(bww1gt_1,173,591). coords(bww1gt_1,100,249,591).  visible(bww1gt_1,591). happensAt(inactive(bww1gt_1),591). orientation(bww1gt_2,173,591). coords(bww1gt_2,135,211,591).  visible(bww1gt_2,591). happensAt(walking(bww1gt_2),591).
orientation(bww1gt_1,173,592). coords(bww1gt_1,100,249,592).  visible(bww1gt_1,592). happensAt(walking(bww1gt_1),592). orientation(bww1gt_2,177,592). coords(bww1gt_2,135,211,592).  visible(bww1gt_2,592). happensAt(walking(bww1gt_2),592).
orientation(bww1gt_1,173,593). coords(bww1gt_1,100,249,593).  visible(bww1gt_1,593). happensAt(walking(bww1gt_1),593). orientation(bww1gt_2,177,593). coords(bww1gt_2,134,211,593).  visible(bww1gt_2,593). happensAt(walking(bww1gt_2),593).
orientation(bww1gt_1,173,594). coords(bww1gt_1,100,249,594).  visible(bww1gt_1,594). happensAt(walking(bww1gt_1),594). orientation(bww1gt_2,175,594). coords(bww1gt_2,134,211,594).  visible(bww1gt_2,594). happensAt(walking(bww1gt_2),594).
orientation(bww1gt_1,173,595). coords(bww1gt_1,100,249,595).  visible(bww1gt_1,595). happensAt(walking(bww1gt_1),595). orientation(bww1gt_2,177,595). coords(bww1gt_2,134,211,595).  visible(bww1gt_2,595). happensAt(walking(bww1gt_2),595).
orientation(bww1gt_1,173,596). coords(bww1gt_1,100,249,596).  visible(bww1gt_1,596). happensAt(walking(bww1gt_1),596). orientation(bww1gt_2,1,596). coords(bww1gt_2,134,211,596).  visible(bww1gt_2,596). happensAt(walking(bww1gt_2),596).
orientation(bww1gt_1,173,597). coords(bww1gt_1,100,249,597).  visible(bww1gt_1,597). happensAt(walking(bww1gt_1),597). orientation(bww1gt_2,0,597). coords(bww1gt_2,134,211,597).  visible(bww1gt_2,597). happensAt(walking(bww1gt_2),597).
orientation(bww1gt_1,173,598). coords(bww1gt_1,100,249,598).  visible(bww1gt_1,598). happensAt(walking(bww1gt_1),598). orientation(bww1gt_2,178,598). coords(bww1gt_2,134,211,598).  visible(bww1gt_2,598). happensAt(walking(bww1gt_2),598).
orientation(bww1gt_1,173,599). coords(bww1gt_1,100,249,599).  visible(bww1gt_1,599). happensAt(walking(bww1gt_1),599). orientation(bww1gt_2,1,599). coords(bww1gt_2,134,211,599).  visible(bww1gt_2,599). happensAt(active(bww1gt_2),599).
orientation(bww1gt_1,173,600). coords(bww1gt_1,101,250,600).  visible(bww1gt_1,600). happensAt(walking(bww1gt_1),600). orientation(bww1gt_2,178,600). coords(bww1gt_2,134,211,600).  visible(bww1gt_2,600). happensAt(active(bww1gt_2),600).
orientation(bww1gt_1,173,601). coords(bww1gt_1,101,250,601).  visible(bww1gt_1,601). happensAt(walking(bww1gt_1),601). orientation(bww1gt_2,1,601). coords(bww1gt_2,134,211,601).  visible(bww1gt_2,601). happensAt(active(bww1gt_2),601).
orientation(bww1gt_1,173,602). coords(bww1gt_1,101,250,602).  visible(bww1gt_1,602). happensAt(walking(bww1gt_1),602). orientation(bww1gt_2,0,602). coords(bww1gt_2,134,211,602).  visible(bww1gt_2,602). happensAt(active(bww1gt_2),602).
orientation(bww1gt_1,173,603). coords(bww1gt_1,101,250,603).  visible(bww1gt_1,603). happensAt(walking(bww1gt_1),603). orientation(bww1gt_2,178,603). coords(bww1gt_2,134,211,603).  visible(bww1gt_2,603). happensAt(active(bww1gt_2),603).
orientation(bww1gt_1,173,604). coords(bww1gt_1,101,250,604).  visible(bww1gt_1,604). happensAt(walking(bww1gt_1),604). orientation(bww1gt_2,1,604). coords(bww1gt_2,134,211,604).  visible(bww1gt_2,604). happensAt(active(bww1gt_2),604).
orientation(bww1gt_1,173,605). coords(bww1gt_1,101,250,605).  visible(bww1gt_1,605). happensAt(walking(bww1gt_1),605). orientation(bww1gt_2,178,605). coords(bww1gt_2,135,211,605).  visible(bww1gt_2,605). happensAt(active(bww1gt_2),605).
orientation(bww1gt_1,173,606). coords(bww1gt_1,101,250,606).  visible(bww1gt_1,606). happensAt(walking(bww1gt_1),606). orientation(bww1gt_2,179,606). coords(bww1gt_2,135,211,606).  visible(bww1gt_2,606). happensAt(active(bww1gt_2),606).
orientation(bww1gt_1,173,607). coords(bww1gt_1,101,250,607).  visible(bww1gt_1,607). happensAt(walking(bww1gt_1),607). orientation(bww1gt_2,176,607). coords(bww1gt_2,135,211,607).  visible(bww1gt_2,607). happensAt(active(bww1gt_2),607).
orientation(bww1gt_1,173,608). coords(bww1gt_1,101,250,608).  visible(bww1gt_1,608). happensAt(walking(bww1gt_1),608). orientation(bww1gt_2,177,608). coords(bww1gt_2,135,212,608).  visible(bww1gt_2,608). happensAt(active(bww1gt_2),608).
orientation(bww1gt_1,173,609). coords(bww1gt_1,101,250,609).  visible(bww1gt_1,609). happensAt(walking(bww1gt_1),609). orientation(bww1gt_2,177,609). coords(bww1gt_2,136,212,609).  visible(bww1gt_2,609). happensAt(active(bww1gt_2),609).
orientation(bww1gt_1,173,610). coords(bww1gt_1,101,251,610).  visible(bww1gt_1,610). happensAt(walking(bww1gt_1),610). orientation(bww1gt_2,1,610). coords(bww1gt_2,137,212,610).  visible(bww1gt_2,610). happensAt(active(bww1gt_2),610).
orientation(bww1gt_1,173,611). coords(bww1gt_1,101,251,611).  visible(bww1gt_1,611). happensAt(walking(bww1gt_1),611). orientation(bww1gt_2,2,611). coords(bww1gt_2,138,211,611).  visible(bww1gt_2,611). happensAt(active(bww1gt_2),611).
orientation(bww1gt_1,173,612). coords(bww1gt_1,101,251,612).  visible(bww1gt_1,612). happensAt(walking(bww1gt_1),612). orientation(bww1gt_2,179,612). coords(bww1gt_2,139,211,612).  visible(bww1gt_2,612). happensAt(active(bww1gt_2),612).
orientation(bww1gt_1,173,613). coords(bww1gt_1,101,251,613).  visible(bww1gt_1,613). happensAt(walking(bww1gt_1),613). orientation(bww1gt_2,1,613). coords(bww1gt_2,139,210,613).  visible(bww1gt_2,613). happensAt(active(bww1gt_2),613).
orientation(bww1gt_1,173,614). coords(bww1gt_1,101,251,614).  visible(bww1gt_1,614). happensAt(walking(bww1gt_1),614). orientation(bww1gt_2,2,614). coords(bww1gt_2,140,209,614).  visible(bww1gt_2,614). happensAt(active(bww1gt_2),614).
#show holdsAt/2.