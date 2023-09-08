
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

	orientation(fomdgt3_2,130,579). coords(fomdgt3_2,248,95,579).  visible(fomdgt3_2,579). happensAt(walking(fomdgt3_2),579). orientation(fomdgt3_3,159,579). coords(fomdgt3_3,50,101,579).  visible(fomdgt3_3,579). happensAt(walking(fomdgt3_3),579).
orientation(fomdgt3_2,130,580). coords(fomdgt3_2,249,97,580).  visible(fomdgt3_2,580). happensAt(walking(fomdgt3_2),580). orientation(fomdgt3_3,159,580). coords(fomdgt3_3,50,101,580).  visible(fomdgt3_3,580). happensAt(walking(fomdgt3_3),580).
orientation(fomdgt3_2,130,581). coords(fomdgt3_2,249,98,581).  visible(fomdgt3_2,581). happensAt(walking(fomdgt3_2),581). orientation(fomdgt3_3,159,581). coords(fomdgt3_3,50,101,581).  visible(fomdgt3_3,581). happensAt(walking(fomdgt3_3),581).
orientation(fomdgt3_2,130,582). coords(fomdgt3_2,249,99,582).  visible(fomdgt3_2,582). happensAt(walking(fomdgt3_2),582). orientation(fomdgt3_3,159,582). coords(fomdgt3_3,50,101,582).  visible(fomdgt3_3,582). happensAt(walking(fomdgt3_3),582).
orientation(fomdgt3_2,130,583). coords(fomdgt3_2,248,101,583).  visible(fomdgt3_2,583). happensAt(walking(fomdgt3_2),583). orientation(fomdgt3_3,159,583). coords(fomdgt3_3,52,100,583).  visible(fomdgt3_3,583). happensAt(walking(fomdgt3_3),583).
orientation(fomdgt3_2,130,584). coords(fomdgt3_2,249,101,584).  visible(fomdgt3_2,584). happensAt(walking(fomdgt3_2),584). orientation(fomdgt3_3,159,584). coords(fomdgt3_3,52,100,584).  visible(fomdgt3_3,584). happensAt(walking(fomdgt3_3),584).
orientation(fomdgt3_2,130,585). coords(fomdgt3_2,249,101,585).  visible(fomdgt3_2,585). happensAt(walking(fomdgt3_2),585). orientation(fomdgt3_3,159,585). coords(fomdgt3_3,52,100,585).  visible(fomdgt3_3,585). happensAt(walking(fomdgt3_3),585).
orientation(fomdgt3_2,130,586). coords(fomdgt3_2,249,102,586).  visible(fomdgt3_2,586). happensAt(walking(fomdgt3_2),586). orientation(fomdgt3_3,159,586). coords(fomdgt3_3,52,100,586).  visible(fomdgt3_3,586). happensAt(walking(fomdgt3_3),586).
orientation(fomdgt3_2,130,587). coords(fomdgt3_2,250,103,587).  visible(fomdgt3_2,587). happensAt(walking(fomdgt3_2),587). orientation(fomdgt3_3,159,587). coords(fomdgt3_3,52,100,587).  visible(fomdgt3_3,587). happensAt(walking(fomdgt3_3),587).
orientation(fomdgt3_2,130,588). coords(fomdgt3_2,250,103,588).  visible(fomdgt3_2,588). happensAt(walking(fomdgt3_2),588). orientation(fomdgt3_3,159,588). coords(fomdgt3_3,53,100,588).  visible(fomdgt3_3,588). happensAt(walking(fomdgt3_3),588).
orientation(fomdgt3_2,130,589). coords(fomdgt3_2,250,104,589).  visible(fomdgt3_2,589). happensAt(walking(fomdgt3_2),589). orientation(fomdgt3_3,159,589). coords(fomdgt3_3,55,99,589).  visible(fomdgt3_3,589). happensAt(walking(fomdgt3_3),589).
orientation(fomdgt3_2,130,590). coords(fomdgt3_2,250,105,590).  visible(fomdgt3_2,590). happensAt(walking(fomdgt3_2),590). orientation(fomdgt3_3,159,590). coords(fomdgt3_3,55,99,590).  visible(fomdgt3_3,590). happensAt(walking(fomdgt3_3),590).
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
#show holdsAt/2.