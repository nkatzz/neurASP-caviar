
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

	orientation(fcgt_2,150,282). coords(fcgt_2,78,137,282).  visible(fcgt_2,282). happensAt(walking(fcgt_2),282). orientation(fcgt_3,164,282). coords(fcgt_3,98,185,282).  visible(fcgt_3,282). happensAt(walking(fcgt_3),282).
orientation(fcgt_2,150,283). coords(fcgt_2,81,136,283).  visible(fcgt_2,283). happensAt(walking(fcgt_2),283). orientation(fcgt_3,164,283). coords(fcgt_3,98,185,283).  visible(fcgt_3,283). happensAt(walking(fcgt_3),283).
orientation(fcgt_2,150,284). coords(fcgt_2,82,136,284).  visible(fcgt_2,284). happensAt(walking(fcgt_2),284). orientation(fcgt_3,164,284). coords(fcgt_3,98,185,284).  visible(fcgt_3,284). happensAt(walking(fcgt_3),284).
orientation(fcgt_2,150,285). coords(fcgt_2,82,136,285).  visible(fcgt_2,285). happensAt(walking(fcgt_2),285). orientation(fcgt_3,164,285). coords(fcgt_3,98,185,285).  visible(fcgt_3,285). happensAt(walking(fcgt_3),285).
orientation(fcgt_2,150,286). coords(fcgt_2,82,135,286).  visible(fcgt_2,286). happensAt(walking(fcgt_2),286). orientation(fcgt_3,164,286). coords(fcgt_3,98,185,286).  visible(fcgt_3,286). happensAt(walking(fcgt_3),286).
orientation(fcgt_2,150,287). coords(fcgt_2,83,135,287).  visible(fcgt_2,287). happensAt(walking(fcgt_2),287). orientation(fcgt_3,164,287). coords(fcgt_3,98,185,287).  visible(fcgt_3,287). happensAt(walking(fcgt_3),287).
orientation(fcgt_2,150,288). coords(fcgt_2,83,134,288).  visible(fcgt_2,288). happensAt(walking(fcgt_2),288). orientation(fcgt_3,164,288). coords(fcgt_3,98,185,288).  visible(fcgt_3,288). happensAt(inactive(fcgt_3),288).
orientation(fcgt_2,150,289). coords(fcgt_2,83,134,289).  visible(fcgt_2,289). happensAt(walking(fcgt_2),289). orientation(fcgt_3,164,289). coords(fcgt_3,98,186,289).  visible(fcgt_3,289). happensAt(inactive(fcgt_3),289).
orientation(fcgt_2,150,290). coords(fcgt_2,84,133,290).  visible(fcgt_2,290). happensAt(walking(fcgt_2),290). orientation(fcgt_3,164,290). coords(fcgt_3,98,186,290).  visible(fcgt_3,290). happensAt(inactive(fcgt_3),290).
orientation(fcgt_2,150,291). coords(fcgt_2,84,133,291).  visible(fcgt_2,291). happensAt(walking(fcgt_2),291). orientation(fcgt_3,164,291). coords(fcgt_3,98,186,291).  visible(fcgt_3,291). happensAt(inactive(fcgt_3),291).
orientation(fcgt_2,150,292). coords(fcgt_2,85,133,292).  visible(fcgt_2,292). happensAt(walking(fcgt_2),292). orientation(fcgt_3,164,292). coords(fcgt_3,98,186,292).  visible(fcgt_3,292). happensAt(inactive(fcgt_3),292).
orientation(fcgt_2,150,293). coords(fcgt_2,85,133,293).  visible(fcgt_2,293). happensAt(walking(fcgt_2),293). orientation(fcgt_3,164,293). coords(fcgt_3,98,186,293).  visible(fcgt_3,293). happensAt(inactive(fcgt_3),293).
orientation(fcgt_2,150,294). coords(fcgt_2,86,132,294).  visible(fcgt_2,294). happensAt(walking(fcgt_2),294). orientation(fcgt_3,164,294). coords(fcgt_3,98,186,294).  visible(fcgt_3,294). happensAt(inactive(fcgt_3),294).
orientation(fcgt_2,150,295). coords(fcgt_2,86,132,295).  visible(fcgt_2,295). happensAt(walking(fcgt_2),295). orientation(fcgt_3,164,295). coords(fcgt_3,98,186,295).  visible(fcgt_3,295). happensAt(inactive(fcgt_3),295).
orientation(fcgt_2,150,296). coords(fcgt_2,86,131,296).  visible(fcgt_2,296). happensAt(walking(fcgt_2),296). orientation(fcgt_3,164,296). coords(fcgt_3,98,186,296).  visible(fcgt_3,296). happensAt(inactive(fcgt_3),296).
orientation(fcgt_2,150,297). coords(fcgt_2,86,131,297).  visible(fcgt_2,297). happensAt(active(fcgt_2),297). orientation(fcgt_3,164,297). coords(fcgt_3,98,187,297).  visible(fcgt_3,297). happensAt(inactive(fcgt_3),297).
orientation(fcgt_2,150,298). coords(fcgt_2,87,131,298).  visible(fcgt_2,298). happensAt(active(fcgt_2),298). orientation(fcgt_3,164,298). coords(fcgt_3,98,187,298).  visible(fcgt_3,298). happensAt(inactive(fcgt_3),298).
orientation(fcgt_2,150,299). coords(fcgt_2,88,131,299).  visible(fcgt_2,299). happensAt(active(fcgt_2),299). orientation(fcgt_3,164,299). coords(fcgt_3,98,187,299).  visible(fcgt_3,299). happensAt(inactive(fcgt_3),299).
orientation(fcgt_2,150,300). coords(fcgt_2,90,130,300).  visible(fcgt_2,300). happensAt(active(fcgt_2),300). orientation(fcgt_3,164,300). coords(fcgt_3,98,187,300).  visible(fcgt_3,300). happensAt(inactive(fcgt_3),300).
orientation(fcgt_2,150,301). coords(fcgt_2,91,130,301).  visible(fcgt_2,301). happensAt(active(fcgt_2),301). orientation(fcgt_3,164,301). coords(fcgt_3,98,187,301).  visible(fcgt_3,301). happensAt(inactive(fcgt_3),301).
orientation(fcgt_2,150,302). coords(fcgt_2,91,130,302).  visible(fcgt_2,302). happensAt(active(fcgt_2),302). orientation(fcgt_3,164,302). coords(fcgt_3,98,187,302).  visible(fcgt_3,302). happensAt(inactive(fcgt_3),302).
orientation(fcgt_2,150,303). coords(fcgt_2,92,130,303).  visible(fcgt_2,303). happensAt(active(fcgt_2),303). orientation(fcgt_3,164,303). coords(fcgt_3,98,187,303).  visible(fcgt_3,303). happensAt(inactive(fcgt_3),303).
orientation(fcgt_2,150,304). coords(fcgt_2,92,130,304).  visible(fcgt_2,304). happensAt(active(fcgt_2),304). orientation(fcgt_3,164,304). coords(fcgt_3,98,187,304).  visible(fcgt_3,304). happensAt(inactive(fcgt_3),304).
orientation(fcgt_2,150,305). coords(fcgt_2,92,130,305).  visible(fcgt_2,305). happensAt(active(fcgt_2),305). orientation(fcgt_3,164,305). coords(fcgt_3,98,187,305).  visible(fcgt_3,305). happensAt(inactive(fcgt_3),305).
orientation(fcgt_2,150,306). coords(fcgt_2,92,129,306).  visible(fcgt_2,306). happensAt(active(fcgt_2),306). orientation(fcgt_3,164,306). coords(fcgt_3,98,187,306).  visible(fcgt_3,306). happensAt(inactive(fcgt_3),306).
#show holdsAt/2.