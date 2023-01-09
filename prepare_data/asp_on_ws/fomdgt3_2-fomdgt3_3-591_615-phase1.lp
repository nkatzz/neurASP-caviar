
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

	orientation(fomdgt3_2,130,591). coords(fomdgt3_2,250,106,591).  visible(fomdgt3_2,591). happensAt(walking(fomdgt3_2),591). orientation(fomdgt3_3,159,591). coords(fomdgt3_3,57,99,591).  visible(fomdgt3_3,591). happensAt(walking(fomdgt3_3),591).
orientation(fomdgt3_2,130,592). coords(fomdgt3_2,251,107,592).  visible(fomdgt3_2,592). happensAt(walking(fomdgt3_2),592). orientation(fomdgt3_3,159,592). coords(fomdgt3_3,57,100,592).  visible(fomdgt3_3,592). happensAt(walking(fomdgt3_3),592).
orientation(fomdgt3_2,130,593). coords(fomdgt3_2,250,107,593).  visible(fomdgt3_2,593). happensAt(walking(fomdgt3_2),593). orientation(fomdgt3_3,159,593). coords(fomdgt3_3,57,99,593).  visible(fomdgt3_3,593). happensAt(walking(fomdgt3_3),593).
orientation(fomdgt3_2,130,594). coords(fomdgt3_2,251,108,594).  visible(fomdgt3_2,594). happensAt(walking(fomdgt3_2),594). orientation(fomdgt3_3,159,594). coords(fomdgt3_3,57,98,594).  visible(fomdgt3_3,594). happensAt(walking(fomdgt3_3),594).
orientation(fomdgt3_2,130,595). coords(fomdgt3_2,252,108,595).  visible(fomdgt3_2,595). happensAt(walking(fomdgt3_2),595). orientation(fomdgt3_3,159,595). coords(fomdgt3_3,57,98,595).  visible(fomdgt3_3,595). happensAt(walking(fomdgt3_3),595).
orientation(fomdgt3_2,130,596). coords(fomdgt3_2,252,109,596).  visible(fomdgt3_2,596). happensAt(walking(fomdgt3_2),596). orientation(fomdgt3_3,159,596). coords(fomdgt3_3,58,98,596).  visible(fomdgt3_3,596). happensAt(walking(fomdgt3_3),596).
orientation(fomdgt3_2,130,597). coords(fomdgt3_2,252,109,597).  visible(fomdgt3_2,597). happensAt(walking(fomdgt3_2),597). orientation(fomdgt3_3,159,597). coords(fomdgt3_3,59,98,597).  visible(fomdgt3_3,597). happensAt(walking(fomdgt3_3),597).
orientation(fomdgt3_2,130,598). coords(fomdgt3_2,252,109,598).  visible(fomdgt3_2,598). happensAt(walking(fomdgt3_2),598). orientation(fomdgt3_3,159,598). coords(fomdgt3_3,59,98,598).  visible(fomdgt3_3,598). happensAt(walking(fomdgt3_3),598).
orientation(fomdgt3_2,130,599). coords(fomdgt3_2,252,109,599).  visible(fomdgt3_2,599). happensAt(walking(fomdgt3_2),599). orientation(fomdgt3_3,159,599). coords(fomdgt3_3,60,98,599).  visible(fomdgt3_3,599). happensAt(walking(fomdgt3_3),599).
orientation(fomdgt3_2,130,600). coords(fomdgt3_2,252,109,600).  visible(fomdgt3_2,600). happensAt(walking(fomdgt3_2),600). orientation(fomdgt3_3,159,600). coords(fomdgt3_3,60,98,600).  visible(fomdgt3_3,600). happensAt(walking(fomdgt3_3),600).
orientation(fomdgt3_2,130,601). coords(fomdgt3_2,253,111,601).  visible(fomdgt3_2,601). happensAt(walking(fomdgt3_2),601). orientation(fomdgt3_3,159,601). coords(fomdgt3_3,60,98,601).  visible(fomdgt3_3,601). happensAt(walking(fomdgt3_3),601).
orientation(fomdgt3_2,130,602). coords(fomdgt3_2,253,113,602).  visible(fomdgt3_2,602). happensAt(walking(fomdgt3_2),602). orientation(fomdgt3_3,159,602). coords(fomdgt3_3,60,98,602).  visible(fomdgt3_3,602). happensAt(walking(fomdgt3_3),602).
orientation(fomdgt3_2,130,603). coords(fomdgt3_2,253,117,603).  visible(fomdgt3_2,603). happensAt(walking(fomdgt3_2),603). orientation(fomdgt3_3,159,603). coords(fomdgt3_3,60,98,603).  visible(fomdgt3_3,603). happensAt(walking(fomdgt3_3),603).
orientation(fomdgt3_2,130,604). coords(fomdgt3_2,253,119,604).  visible(fomdgt3_2,604). happensAt(walking(fomdgt3_2),604). orientation(fomdgt3_3,159,604). coords(fomdgt3_3,60,98,604).  visible(fomdgt3_3,604). happensAt(walking(fomdgt3_3),604).
orientation(fomdgt3_2,130,605). coords(fomdgt3_2,253,122,605).  visible(fomdgt3_2,605). happensAt(walking(fomdgt3_2),605). orientation(fomdgt3_3,159,605). coords(fomdgt3_3,60,98,605).  visible(fomdgt3_3,605). happensAt(walking(fomdgt3_3),605).
orientation(fomdgt3_2,130,606). coords(fomdgt3_2,253,125,606).  visible(fomdgt3_2,606). happensAt(walking(fomdgt3_2),606). orientation(fomdgt3_3,159,606). coords(fomdgt3_3,63,98,606).  visible(fomdgt3_3,606). happensAt(walking(fomdgt3_3),606).
orientation(fomdgt3_2,130,607). coords(fomdgt3_2,253,127,607).  visible(fomdgt3_2,607). happensAt(walking(fomdgt3_2),607). orientation(fomdgt3_3,159,607). coords(fomdgt3_3,64,98,607).  visible(fomdgt3_3,607). happensAt(walking(fomdgt3_3),607).
orientation(fomdgt3_2,130,608). coords(fomdgt3_2,253,128,608).  visible(fomdgt3_2,608). happensAt(walking(fomdgt3_2),608). orientation(fomdgt3_3,159,608). coords(fomdgt3_3,65,97,608).  visible(fomdgt3_3,608). happensAt(walking(fomdgt3_3),608).
orientation(fomdgt3_2,130,609). coords(fomdgt3_2,253,130,609).  visible(fomdgt3_2,609). happensAt(walking(fomdgt3_2),609). orientation(fomdgt3_3,159,609). coords(fomdgt3_3,65,96,609).  visible(fomdgt3_3,609). happensAt(walking(fomdgt3_3),609).
orientation(fomdgt3_2,130,610). coords(fomdgt3_2,253,131,610).  visible(fomdgt3_2,610). happensAt(walking(fomdgt3_2),610). orientation(fomdgt3_3,159,610). coords(fomdgt3_3,65,96,610).  visible(fomdgt3_3,610). happensAt(walking(fomdgt3_3),610).
orientation(fomdgt3_2,130,611). coords(fomdgt3_2,253,132,611).  visible(fomdgt3_2,611). happensAt(walking(fomdgt3_2),611). orientation(fomdgt3_3,159,611). coords(fomdgt3_3,65,96,611).  visible(fomdgt3_3,611). happensAt(walking(fomdgt3_3),611).
orientation(fomdgt3_2,130,612). coords(fomdgt3_2,253,133,612).  visible(fomdgt3_2,612). happensAt(walking(fomdgt3_2),612). orientation(fomdgt3_3,159,612). coords(fomdgt3_3,65,96,612).  visible(fomdgt3_3,612). happensAt(walking(fomdgt3_3),612).
orientation(fomdgt3_2,130,613). coords(fomdgt3_2,253,135,613).  visible(fomdgt3_2,613). happensAt(walking(fomdgt3_2),613). orientation(fomdgt3_3,159,613). coords(fomdgt3_3,66,94,613).  visible(fomdgt3_3,613). happensAt(walking(fomdgt3_3),613).
orientation(fomdgt3_2,130,614). coords(fomdgt3_2,253,135,614).  visible(fomdgt3_2,614). happensAt(walking(fomdgt3_2),614). orientation(fomdgt3_3,159,614). coords(fomdgt3_3,66,94,614).  visible(fomdgt3_3,614). happensAt(walking(fomdgt3_3),614).
orientation(fomdgt3_2,130,615). coords(fomdgt3_2,253,135,615).  visible(fomdgt3_2,615). happensAt(walking(fomdgt3_2),615). orientation(fomdgt3_3,159,615). coords(fomdgt3_3,69,95,615).  visible(fomdgt3_3,615). happensAt(walking(fomdgt3_3),615).
#show holdsAt/2.