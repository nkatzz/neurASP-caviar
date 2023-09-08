
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

	orientation(fomdgt3_1,150,282). coords(fomdgt3_1,51,96,282).  visible(fomdgt3_1,282). happensAt(walking(fomdgt3_1),282). orientation(fomdgt3_6,130,282). coords(fomdgt3_6,219,152,282).  visible(fomdgt3_6,282). happensAt(walking(fomdgt3_6),282).
orientation(fomdgt3_1,150,283). coords(fomdgt3_1,51,96,283).  visible(fomdgt3_1,283). happensAt(walking(fomdgt3_1),283). orientation(fomdgt3_6,130,283). coords(fomdgt3_6,223,164,283).  visible(fomdgt3_6,283). happensAt(walking(fomdgt3_6),283).
orientation(fomdgt3_1,150,284). coords(fomdgt3_1,51,96,284).  visible(fomdgt3_1,284). happensAt(walking(fomdgt3_1),284). orientation(fomdgt3_6,130,284). coords(fomdgt3_6,216,155,284).  visible(fomdgt3_6,284). happensAt(walking(fomdgt3_6),284).
orientation(fomdgt3_1,150,285). coords(fomdgt3_1,51,96,285).  visible(fomdgt3_1,285). happensAt(walking(fomdgt3_1),285). orientation(fomdgt3_6,130,285). coords(fomdgt3_6,216,151,285).  visible(fomdgt3_6,285). happensAt(walking(fomdgt3_6),285).
orientation(fomdgt3_1,150,286). coords(fomdgt3_1,51,96,286).  visible(fomdgt3_1,286). happensAt(active(fomdgt3_1),286). orientation(fomdgt3_6,130,286). coords(fomdgt3_6,216,148,286).  visible(fomdgt3_6,286). happensAt(walking(fomdgt3_6),286).
orientation(fomdgt3_1,150,287). coords(fomdgt3_1,51,96,287).  visible(fomdgt3_1,287). happensAt(active(fomdgt3_1),287). orientation(fomdgt3_6,130,287). coords(fomdgt3_6,218,144,287).  visible(fomdgt3_6,287). happensAt(walking(fomdgt3_6),287).
orientation(fomdgt3_1,150,288). coords(fomdgt3_1,51,96,288).  visible(fomdgt3_1,288). happensAt(active(fomdgt3_1),288). orientation(fomdgt3_6,130,288). coords(fomdgt3_6,218,143,288).  visible(fomdgt3_6,288). happensAt(walking(fomdgt3_6),288).
orientation(fomdgt3_1,150,289). coords(fomdgt3_1,46,96,289).  visible(fomdgt3_1,289). happensAt(active(fomdgt3_1),289). orientation(fomdgt3_6,130,289). coords(fomdgt3_6,219,141,289).  visible(fomdgt3_6,289). happensAt(walking(fomdgt3_6),289).
orientation(fomdgt3_1,150,290). coords(fomdgt3_1,46,96,290).  visible(fomdgt3_1,290). happensAt(active(fomdgt3_1),290). orientation(fomdgt3_6,130,290). coords(fomdgt3_6,219,141,290).  visible(fomdgt3_6,290). happensAt(walking(fomdgt3_6),290).
orientation(fomdgt3_1,150,291). coords(fomdgt3_1,46,96,291).  visible(fomdgt3_1,291). happensAt(active(fomdgt3_1),291). orientation(fomdgt3_6,130,291). coords(fomdgt3_6,219,141,291).  visible(fomdgt3_6,291). happensAt(walking(fomdgt3_6),291).
orientation(fomdgt3_1,150,292). coords(fomdgt3_1,46,96,292).  visible(fomdgt3_1,292). happensAt(active(fomdgt3_1),292). orientation(fomdgt3_6,130,292). coords(fomdgt3_6,221,138,292).  visible(fomdgt3_6,292). happensAt(walking(fomdgt3_6),292).
orientation(fomdgt3_1,150,293). coords(fomdgt3_1,46,96,293).  visible(fomdgt3_1,293). happensAt(active(fomdgt3_1),293). orientation(fomdgt3_6,130,293). coords(fomdgt3_6,219,136,293).  visible(fomdgt3_6,293). happensAt(walking(fomdgt3_6),293).
orientation(fomdgt3_1,150,294). coords(fomdgt3_1,46,96,294).  visible(fomdgt3_1,294). happensAt(active(fomdgt3_1),294). orientation(fomdgt3_6,130,294). coords(fomdgt3_6,217,133,294).  visible(fomdgt3_6,294). happensAt(walking(fomdgt3_6),294).
orientation(fomdgt3_1,150,295). coords(fomdgt3_1,46,96,295).  visible(fomdgt3_1,295). happensAt(active(fomdgt3_1),295). orientation(fomdgt3_6,130,295). coords(fomdgt3_6,217,130,295).  visible(fomdgt3_6,295). happensAt(walking(fomdgt3_6),295).
orientation(fomdgt3_1,150,296). coords(fomdgt3_1,46,96,296).  visible(fomdgt3_1,296). happensAt(inactive(fomdgt3_1),296). orientation(fomdgt3_6,130,296). coords(fomdgt3_6,217,127,296).  visible(fomdgt3_6,296). happensAt(walking(fomdgt3_6),296).
orientation(fomdgt3_1,150,297). coords(fomdgt3_1,46,96,297).  visible(fomdgt3_1,297). happensAt(inactive(fomdgt3_1),297). orientation(fomdgt3_6,130,297). coords(fomdgt3_6,217,125,297).  visible(fomdgt3_6,297). happensAt(walking(fomdgt3_6),297).
orientation(fomdgt3_1,150,298). coords(fomdgt3_1,46,96,298).  visible(fomdgt3_1,298). happensAt(inactive(fomdgt3_1),298). orientation(fomdgt3_6,130,298). coords(fomdgt3_6,217,124,298).  visible(fomdgt3_6,298). happensAt(walking(fomdgt3_6),298).
orientation(fomdgt3_1,150,299). coords(fomdgt3_1,46,96,299).  visible(fomdgt3_1,299). happensAt(inactive(fomdgt3_1),299). orientation(fomdgt3_6,130,299). coords(fomdgt3_6,217,123,299).  visible(fomdgt3_6,299). happensAt(walking(fomdgt3_6),299).
orientation(fomdgt3_1,150,300). coords(fomdgt3_1,46,96,300).  visible(fomdgt3_1,300). happensAt(inactive(fomdgt3_1),300). orientation(fomdgt3_6,130,300). coords(fomdgt3_6,217,123,300).  visible(fomdgt3_6,300). happensAt(walking(fomdgt3_6),300).
orientation(fomdgt3_1,150,301). coords(fomdgt3_1,46,96,301).  visible(fomdgt3_1,301). happensAt(inactive(fomdgt3_1),301). orientation(fomdgt3_6,130,301). coords(fomdgt3_6,216,120,301).  visible(fomdgt3_6,301). happensAt(walking(fomdgt3_6),301).
orientation(fomdgt3_1,150,302). coords(fomdgt3_1,46,96,302).  visible(fomdgt3_1,302). happensAt(inactive(fomdgt3_1),302). orientation(fomdgt3_6,130,302). coords(fomdgt3_6,215,120,302).  visible(fomdgt3_6,302). happensAt(walking(fomdgt3_6),302).
orientation(fomdgt3_1,150,303). coords(fomdgt3_1,54,100,303).  visible(fomdgt3_1,303). happensAt(inactive(fomdgt3_1),303). orientation(fomdgt3_6,130,303). coords(fomdgt3_6,215,119,303).  visible(fomdgt3_6,303). happensAt(walking(fomdgt3_6),303).
orientation(fomdgt3_1,150,304). coords(fomdgt3_1,54,100,304).  visible(fomdgt3_1,304). happensAt(inactive(fomdgt3_1),304). orientation(fomdgt3_6,130,304). coords(fomdgt3_6,213,115,304).  visible(fomdgt3_6,304). happensAt(walking(fomdgt3_6),304).
orientation(fomdgt3_1,150,305). coords(fomdgt3_1,54,100,305).  visible(fomdgt3_1,305). happensAt(inactive(fomdgt3_1),305). orientation(fomdgt3_6,130,305). coords(fomdgt3_6,213,115,305).  visible(fomdgt3_6,305). happensAt(walking(fomdgt3_6),305).
orientation(fomdgt3_1,150,306). coords(fomdgt3_1,54,100,306).  visible(fomdgt3_1,306). happensAt(inactive(fomdgt3_1),306). orientation(fomdgt3_6,130,306). coords(fomdgt3_6,213,113,306).  visible(fomdgt3_6,306). happensAt(walking(fomdgt3_6),306).
#show holdsAt/2.