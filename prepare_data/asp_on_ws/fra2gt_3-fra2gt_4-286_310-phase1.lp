
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

	orientation(fra2gt_3,153,286). coords(fra2gt_3,81,124,286).  visible(fra2gt_3,286). happensAt(walking(fra2gt_3),286). orientation(fra2gt_4,155,286). coords(fra2gt_4,120,129,286).  visible(fra2gt_4,286). happensAt(walking(fra2gt_4),286).
orientation(fra2gt_3,153,287). coords(fra2gt_3,82,123,287).  visible(fra2gt_3,287). happensAt(walking(fra2gt_3),287). orientation(fra2gt_4,155,287). coords(fra2gt_4,120,130,287).  visible(fra2gt_4,287). happensAt(walking(fra2gt_4),287).
orientation(fra2gt_3,153,288). coords(fra2gt_3,83,122,288).  visible(fra2gt_3,288). happensAt(walking(fra2gt_3),288). orientation(fra2gt_4,155,288). coords(fra2gt_4,119,132,288).  visible(fra2gt_4,288). happensAt(walking(fra2gt_4),288).
orientation(fra2gt_3,150,289). coords(fra2gt_3,84,122,289).  visible(fra2gt_3,289). happensAt(walking(fra2gt_3),289). orientation(fra2gt_4,155,289). coords(fra2gt_4,117,131,289).  visible(fra2gt_4,289). happensAt(walking(fra2gt_4),289).
orientation(fra2gt_3,150,290). coords(fra2gt_3,84,122,290).  visible(fra2gt_3,290). happensAt(walking(fra2gt_3),290). orientation(fra2gt_4,155,290). coords(fra2gt_4,117,130,290).  visible(fra2gt_4,290). happensAt(walking(fra2gt_4),290).
orientation(fra2gt_3,150,291). coords(fra2gt_3,85,121,291).  visible(fra2gt_3,291). happensAt(walking(fra2gt_3),291). orientation(fra2gt_4,155,291). coords(fra2gt_4,116,129,291).  visible(fra2gt_4,291). happensAt(walking(fra2gt_4),291).
orientation(fra2gt_3,150,292). coords(fra2gt_3,85,121,292).  visible(fra2gt_3,292). happensAt(walking(fra2gt_3),292). orientation(fra2gt_4,155,292). coords(fra2gt_4,115,128,292).  visible(fra2gt_4,292). happensAt(walking(fra2gt_4),292).
orientation(fra2gt_3,150,293). coords(fra2gt_3,84,121,293).  visible(fra2gt_3,293). happensAt(walking(fra2gt_3),293). orientation(fra2gt_4,155,293). coords(fra2gt_4,113,128,293).  visible(fra2gt_4,293). happensAt(walking(fra2gt_4),293).
orientation(fra2gt_3,150,294). coords(fra2gt_3,84,120,294).  visible(fra2gt_3,294). happensAt(walking(fra2gt_3),294). orientation(fra2gt_4,155,294). coords(fra2gt_4,112,128,294).  visible(fra2gt_4,294). happensAt(walking(fra2gt_4),294).
orientation(fra2gt_3,150,295). coords(fra2gt_3,84,120,295).  visible(fra2gt_3,295). happensAt(walking(fra2gt_3),295). orientation(fra2gt_4,155,295). coords(fra2gt_4,112,127,295).  visible(fra2gt_4,295). happensAt(walking(fra2gt_4),295).
orientation(fra2gt_3,150,296). coords(fra2gt_3,84,119,296).  visible(fra2gt_3,296). happensAt(walking(fra2gt_3),296). orientation(fra2gt_4,155,296). coords(fra2gt_4,111,125,296).  visible(fra2gt_4,296). happensAt(walking(fra2gt_4),296).
orientation(fra2gt_3,150,297). coords(fra2gt_3,83,118,297).  visible(fra2gt_3,297). happensAt(walking(fra2gt_3),297). orientation(fra2gt_4,155,297). coords(fra2gt_4,109,123,297).  visible(fra2gt_4,297). happensAt(walking(fra2gt_4),297).
orientation(fra2gt_3,150,298). coords(fra2gt_3,85,117,298).  visible(fra2gt_3,298). happensAt(walking(fra2gt_3),298). orientation(fra2gt_4,155,298). coords(fra2gt_4,108,121,298).  visible(fra2gt_4,298). happensAt(walking(fra2gt_4),298).
orientation(fra2gt_3,150,299). coords(fra2gt_3,83,117,299).  visible(fra2gt_3,299). happensAt(walking(fra2gt_3),299). orientation(fra2gt_4,155,299). coords(fra2gt_4,107,120,299).  visible(fra2gt_4,299). happensAt(walking(fra2gt_4),299).
orientation(fra2gt_3,150,300). coords(fra2gt_3,83,116,300).  visible(fra2gt_3,300). happensAt(walking(fra2gt_3),300). orientation(fra2gt_4,155,300). coords(fra2gt_4,101,120,300).  visible(fra2gt_4,300). happensAt(walking(fra2gt_4),300).
orientation(fra2gt_3,150,301). coords(fra2gt_3,84,116,301).  visible(fra2gt_3,301). happensAt(walking(fra2gt_3),301). orientation(fra2gt_4,155,301). coords(fra2gt_4,101,120,301).  visible(fra2gt_4,301). happensAt(walking(fra2gt_4),301).
orientation(fra2gt_3,150,302). coords(fra2gt_3,83,116,302).  visible(fra2gt_3,302). happensAt(walking(fra2gt_3),302). orientation(fra2gt_4,155,302). coords(fra2gt_4,100,120,302).  visible(fra2gt_4,302). happensAt(walking(fra2gt_4),302).
orientation(fra2gt_3,150,303). coords(fra2gt_3,82,116,303).  visible(fra2gt_3,303). happensAt(walking(fra2gt_3),303). orientation(fra2gt_4,155,303). coords(fra2gt_4,99,120,303).  visible(fra2gt_4,303). happensAt(walking(fra2gt_4),303).
orientation(fra2gt_3,150,304). coords(fra2gt_3,82,116,304).  visible(fra2gt_3,304). happensAt(walking(fra2gt_3),304). orientation(fra2gt_4,155,304). coords(fra2gt_4,100,123,304).  visible(fra2gt_4,304). happensAt(walking(fra2gt_4),304).
orientation(fra2gt_3,150,305). coords(fra2gt_3,84,112,305).  visible(fra2gt_3,305). happensAt(walking(fra2gt_3),305). orientation(fra2gt_4,155,305). coords(fra2gt_4,99,122,305).  visible(fra2gt_4,305). happensAt(walking(fra2gt_4),305).
orientation(fra2gt_3,150,306). coords(fra2gt_3,93,117,306).  visible(fra2gt_3,306). happensAt(walking(fra2gt_3),306). orientation(fra2gt_4,155,306). coords(fra2gt_4,99,123,306).  visible(fra2gt_4,306). happensAt(walking(fra2gt_4),306).
orientation(fra2gt_3,150,307). coords(fra2gt_3,93,117,307).  visible(fra2gt_3,307). happensAt(walking(fra2gt_3),307). orientation(fra2gt_4,155,307). coords(fra2gt_4,99,123,307).  visible(fra2gt_4,307). happensAt(walking(fra2gt_4),307).
orientation(fra2gt_3,150,308). coords(fra2gt_3,94,117,308).  visible(fra2gt_3,308). happensAt(walking(fra2gt_3),308). orientation(fra2gt_4,155,308). coords(fra2gt_4,98,122,308).  visible(fra2gt_4,308). happensAt(walking(fra2gt_4),308).
orientation(fra2gt_3,150,309). coords(fra2gt_3,95,117,309).  visible(fra2gt_3,309). happensAt(walking(fra2gt_3),309). orientation(fra2gt_4,155,309). coords(fra2gt_4,98,120,309).  visible(fra2gt_4,309). happensAt(walking(fra2gt_4),309).
orientation(fra2gt_3,150,310). coords(fra2gt_3,95,117,310).  visible(fra2gt_3,310). happensAt(walking(fra2gt_3),310). orientation(fra2gt_4,155,310). coords(fra2gt_4,97,119,310).  visible(fra2gt_4,310). happensAt(walking(fra2gt_4),310).
#show holdsAt/2.