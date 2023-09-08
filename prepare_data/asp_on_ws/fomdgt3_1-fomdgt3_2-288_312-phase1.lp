
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

	orientation(fomdgt3_1,150,288). coords(fomdgt3_1,51,96,288).  visible(fomdgt3_1,288). happensAt(active(fomdgt3_1),288). orientation(fomdgt3_2,130,288). coords(fomdgt3_2,224,122,288).  visible(fomdgt3_2,288). happensAt(walking(fomdgt3_2),288).
orientation(fomdgt3_1,150,289). coords(fomdgt3_1,46,96,289).  visible(fomdgt3_1,289). happensAt(active(fomdgt3_1),289). orientation(fomdgt3_2,130,289). coords(fomdgt3_2,224,122,289).  visible(fomdgt3_2,289). happensAt(walking(fomdgt3_2),289).
orientation(fomdgt3_1,150,290). coords(fomdgt3_1,46,96,290).  visible(fomdgt3_1,290). happensAt(active(fomdgt3_1),290). orientation(fomdgt3_2,130,290). coords(fomdgt3_2,224,121,290).  visible(fomdgt3_2,290). happensAt(walking(fomdgt3_2),290).
orientation(fomdgt3_1,150,291). coords(fomdgt3_1,46,96,291).  visible(fomdgt3_1,291). happensAt(active(fomdgt3_1),291). orientation(fomdgt3_2,130,291). coords(fomdgt3_2,224,121,291).  visible(fomdgt3_2,291). happensAt(walking(fomdgt3_2),291).
orientation(fomdgt3_1,150,292). coords(fomdgt3_1,46,96,292).  visible(fomdgt3_1,292). happensAt(active(fomdgt3_1),292). orientation(fomdgt3_2,130,292). coords(fomdgt3_2,223,121,292).  visible(fomdgt3_2,292). happensAt(walking(fomdgt3_2),292).
orientation(fomdgt3_1,150,293). coords(fomdgt3_1,46,96,293).  visible(fomdgt3_1,293). happensAt(active(fomdgt3_1),293). orientation(fomdgt3_2,130,293). coords(fomdgt3_2,223,121,293).  visible(fomdgt3_2,293). happensAt(walking(fomdgt3_2),293).
orientation(fomdgt3_1,150,294). coords(fomdgt3_1,46,96,294).  visible(fomdgt3_1,294). happensAt(active(fomdgt3_1),294). orientation(fomdgt3_2,130,294). coords(fomdgt3_2,223,121,294).  visible(fomdgt3_2,294). happensAt(walking(fomdgt3_2),294).
orientation(fomdgt3_1,150,295). coords(fomdgt3_1,46,96,295).  visible(fomdgt3_1,295). happensAt(active(fomdgt3_1),295). orientation(fomdgt3_2,130,295). coords(fomdgt3_2,223,119,295).  visible(fomdgt3_2,295). happensAt(walking(fomdgt3_2),295).
orientation(fomdgt3_1,150,296). coords(fomdgt3_1,46,96,296).  visible(fomdgt3_1,296). happensAt(inactive(fomdgt3_1),296). orientation(fomdgt3_2,130,296). coords(fomdgt3_2,223,120,296).  visible(fomdgt3_2,296). happensAt(walking(fomdgt3_2),296).
orientation(fomdgt3_1,150,297). coords(fomdgt3_1,46,96,297).  visible(fomdgt3_1,297). happensAt(inactive(fomdgt3_1),297). orientation(fomdgt3_2,110,297). coords(fomdgt3_2,224,120,297).  visible(fomdgt3_2,297). happensAt(walking(fomdgt3_2),297).
orientation(fomdgt3_1,150,298). coords(fomdgt3_1,46,96,298).  visible(fomdgt3_1,298). happensAt(inactive(fomdgt3_1),298). orientation(fomdgt3_2,110,298). coords(fomdgt3_2,224,121,298).  visible(fomdgt3_2,298). happensAt(walking(fomdgt3_2),298).
orientation(fomdgt3_1,150,299). coords(fomdgt3_1,46,96,299).  visible(fomdgt3_1,299). happensAt(inactive(fomdgt3_1),299). orientation(fomdgt3_2,110,299). coords(fomdgt3_2,223,121,299).  visible(fomdgt3_2,299). happensAt(walking(fomdgt3_2),299).
orientation(fomdgt3_1,150,300). coords(fomdgt3_1,46,96,300).  visible(fomdgt3_1,300). happensAt(inactive(fomdgt3_1),300). orientation(fomdgt3_2,110,300). coords(fomdgt3_2,223,121,300).  visible(fomdgt3_2,300). happensAt(walking(fomdgt3_2),300).
orientation(fomdgt3_1,150,301). coords(fomdgt3_1,46,96,301).  visible(fomdgt3_1,301). happensAt(inactive(fomdgt3_1),301). orientation(fomdgt3_2,120,301). coords(fomdgt3_2,230,115,301).  visible(fomdgt3_2,301). happensAt(walking(fomdgt3_2),301).
orientation(fomdgt3_1,150,302). coords(fomdgt3_1,46,96,302).  visible(fomdgt3_1,302). happensAt(inactive(fomdgt3_1),302). orientation(fomdgt3_2,120,302). coords(fomdgt3_2,230,115,302).  visible(fomdgt3_2,302). happensAt(walking(fomdgt3_2),302).
orientation(fomdgt3_1,150,303). coords(fomdgt3_1,54,100,303).  visible(fomdgt3_1,303). happensAt(inactive(fomdgt3_1),303). orientation(fomdgt3_2,120,303). coords(fomdgt3_2,230,115,303).  visible(fomdgt3_2,303). happensAt(walking(fomdgt3_2),303).
orientation(fomdgt3_1,150,304). coords(fomdgt3_1,54,100,304).  visible(fomdgt3_1,304). happensAt(inactive(fomdgt3_1),304). orientation(fomdgt3_2,120,304). coords(fomdgt3_2,232,115,304).  visible(fomdgt3_2,304). happensAt(walking(fomdgt3_2),304).
orientation(fomdgt3_1,150,305). coords(fomdgt3_1,54,100,305).  visible(fomdgt3_1,305). happensAt(inactive(fomdgt3_1),305). orientation(fomdgt3_2,120,305). coords(fomdgt3_2,232,115,305).  visible(fomdgt3_2,305). happensAt(walking(fomdgt3_2),305).
orientation(fomdgt3_1,150,306). coords(fomdgt3_1,54,100,306).  visible(fomdgt3_1,306). happensAt(inactive(fomdgt3_1),306). orientation(fomdgt3_2,120,306). coords(fomdgt3_2,233,116,306).  visible(fomdgt3_2,306). happensAt(walking(fomdgt3_2),306).
orientation(fomdgt3_1,150,307). coords(fomdgt3_1,54,100,307).  visible(fomdgt3_1,307). happensAt(inactive(fomdgt3_1),307). orientation(fomdgt3_2,120,307). coords(fomdgt3_2,233,116,307).  visible(fomdgt3_2,307). happensAt(walking(fomdgt3_2),307).
orientation(fomdgt3_1,150,308). coords(fomdgt3_1,54,100,308).  visible(fomdgt3_1,308). happensAt(inactive(fomdgt3_1),308). orientation(fomdgt3_2,120,308). coords(fomdgt3_2,239,116,308).  visible(fomdgt3_2,308). happensAt(walking(fomdgt3_2),308).
orientation(fomdgt3_1,150,309). coords(fomdgt3_1,54,100,309).  visible(fomdgt3_1,309). happensAt(inactive(fomdgt3_1),309). orientation(fomdgt3_2,120,309). coords(fomdgt3_2,240,116,309).  visible(fomdgt3_2,309). happensAt(walking(fomdgt3_2),309).
orientation(fomdgt3_1,150,310). coords(fomdgt3_1,54,100,310).  visible(fomdgt3_1,310). happensAt(inactive(fomdgt3_1),310). orientation(fomdgt3_2,120,310). coords(fomdgt3_2,240,116,310).  visible(fomdgt3_2,310). happensAt(walking(fomdgt3_2),310).
orientation(fomdgt3_1,150,311). coords(fomdgt3_1,54,100,311).  visible(fomdgt3_1,311). happensAt(inactive(fomdgt3_1),311). orientation(fomdgt3_2,120,311). coords(fomdgt3_2,240,116,311).  visible(fomdgt3_2,311). happensAt(walking(fomdgt3_2),311).
orientation(fomdgt3_1,150,312). coords(fomdgt3_1,54,100,312).  visible(fomdgt3_1,312). happensAt(inactive(fomdgt3_1),312). orientation(fomdgt3_2,120,312). coords(fomdgt3_2,240,116,312).  visible(fomdgt3_2,312). happensAt(walking(fomdgt3_2),312).
#show holdsAt/2.