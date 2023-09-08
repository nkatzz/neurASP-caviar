
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

	orientation(mws1gt_3,155,286). coords(mws1gt_3,80,131,286).  visible(mws1gt_3,286). happensAt(walking(mws1gt_3),286). orientation(mws1gt_5,161,286). coords(mws1gt_5,37,165,286).  visible(mws1gt_5,286). happensAt(walking(mws1gt_5),286).
orientation(mws1gt_3,155,287). coords(mws1gt_3,80,131,287).  visible(mws1gt_3,287). happensAt(walking(mws1gt_3),287). orientation(mws1gt_5,161,287). coords(mws1gt_5,37,166,287).  visible(mws1gt_5,287). happensAt(walking(mws1gt_5),287).
orientation(mws1gt_3,155,288). coords(mws1gt_3,80,131,288).  visible(mws1gt_3,288). happensAt(walking(mws1gt_3),288). orientation(mws1gt_5,161,288). coords(mws1gt_5,37,166,288).  visible(mws1gt_5,288). happensAt(walking(mws1gt_5),288).
orientation(mws1gt_3,155,289). coords(mws1gt_3,80,131,289).  visible(mws1gt_3,289). happensAt(walking(mws1gt_3),289). orientation(mws1gt_5,161,289). coords(mws1gt_5,37,167,289).  visible(mws1gt_5,289). happensAt(walking(mws1gt_5),289).
orientation(mws1gt_3,155,290). coords(mws1gt_3,80,131,290).  visible(mws1gt_3,290). happensAt(walking(mws1gt_3),290). orientation(mws1gt_5,161,290). coords(mws1gt_5,37,168,290).  visible(mws1gt_5,290). happensAt(walking(mws1gt_5),290).
orientation(mws1gt_3,155,291). coords(mws1gt_3,79,131,291).  visible(mws1gt_3,291). happensAt(walking(mws1gt_3),291). orientation(mws1gt_5,168,291). coords(mws1gt_5,37,169,291).  visible(mws1gt_5,291). happensAt(walking(mws1gt_5),291).
orientation(mws1gt_3,155,292). coords(mws1gt_3,79,131,292).  visible(mws1gt_3,292). happensAt(walking(mws1gt_3),292). orientation(mws1gt_5,168,292). coords(mws1gt_5,37,170,292).  visible(mws1gt_5,292). happensAt(walking(mws1gt_5),292).
orientation(mws1gt_3,158,293). coords(mws1gt_3,78,131,293).  visible(mws1gt_3,293). happensAt(walking(mws1gt_3),293). orientation(mws1gt_5,168,293). coords(mws1gt_5,37,171,293).  visible(mws1gt_5,293). happensAt(walking(mws1gt_5),293).
orientation(mws1gt_3,158,294). coords(mws1gt_3,77,131,294).  visible(mws1gt_3,294). happensAt(walking(mws1gt_3),294). orientation(mws1gt_5,168,294). coords(mws1gt_5,37,172,294).  visible(mws1gt_5,294). happensAt(walking(mws1gt_5),294).
orientation(mws1gt_3,158,295). coords(mws1gt_3,76,131,295).  visible(mws1gt_3,295). happensAt(walking(mws1gt_3),295). orientation(mws1gt_5,168,295). coords(mws1gt_5,37,174,295).  visible(mws1gt_5,295). happensAt(walking(mws1gt_5),295).
orientation(mws1gt_3,158,296). coords(mws1gt_3,76,131,296).  visible(mws1gt_3,296). happensAt(walking(mws1gt_3),296). orientation(mws1gt_5,168,296). coords(mws1gt_5,36,175,296).  visible(mws1gt_5,296). happensAt(walking(mws1gt_5),296).
orientation(mws1gt_3,158,297). coords(mws1gt_3,75,131,297).  visible(mws1gt_3,297). happensAt(walking(mws1gt_3),297). orientation(mws1gt_5,168,297). coords(mws1gt_5,36,176,297).  visible(mws1gt_5,297). happensAt(walking(mws1gt_5),297).
orientation(mws1gt_3,158,298). coords(mws1gt_3,75,131,298).  visible(mws1gt_3,298). happensAt(walking(mws1gt_3),298). orientation(mws1gt_5,168,298). coords(mws1gt_5,36,177,298).  visible(mws1gt_5,298). happensAt(walking(mws1gt_5),298).
orientation(mws1gt_3,158,299). coords(mws1gt_3,75,133,299).  visible(mws1gt_3,299). happensAt(walking(mws1gt_3),299). orientation(mws1gt_5,168,299). coords(mws1gt_5,36,178,299).  visible(mws1gt_5,299). happensAt(walking(mws1gt_5),299).
orientation(mws1gt_3,158,300). coords(mws1gt_3,74,134,300).  visible(mws1gt_3,300). happensAt(walking(mws1gt_3),300). orientation(mws1gt_5,168,300). coords(mws1gt_5,36,178,300).  visible(mws1gt_5,300). happensAt(walking(mws1gt_5),300).
orientation(mws1gt_3,158,301). coords(mws1gt_3,73,134,301).  visible(mws1gt_3,301). happensAt(walking(mws1gt_3),301). orientation(mws1gt_5,168,301). coords(mws1gt_5,36,179,301).  visible(mws1gt_5,301). happensAt(walking(mws1gt_5),301).
orientation(mws1gt_3,158,302). coords(mws1gt_3,73,135,302).  visible(mws1gt_3,302). happensAt(walking(mws1gt_3),302). orientation(mws1gt_5,168,302). coords(mws1gt_5,36,179,302).  visible(mws1gt_5,302). happensAt(walking(mws1gt_5),302).
orientation(mws1gt_3,158,303). coords(mws1gt_3,72,135,303).  visible(mws1gt_3,303). happensAt(walking(mws1gt_3),303). orientation(mws1gt_5,168,303). coords(mws1gt_5,36,179,303).  visible(mws1gt_5,303). happensAt(walking(mws1gt_5),303).
orientation(mws1gt_3,158,304). coords(mws1gt_3,72,135,304).  visible(mws1gt_3,304). happensAt(walking(mws1gt_3),304). orientation(mws1gt_5,168,304). coords(mws1gt_5,36,179,304).  visible(mws1gt_5,304). happensAt(walking(mws1gt_5),304).
orientation(mws1gt_3,158,305). coords(mws1gt_3,72,136,305).  visible(mws1gt_3,305). happensAt(walking(mws1gt_3),305). orientation(mws1gt_5,168,305). coords(mws1gt_5,36,180,305).  visible(mws1gt_5,305). happensAt(walking(mws1gt_5),305).
orientation(mws1gt_3,158,306). coords(mws1gt_3,71,136,306).  visible(mws1gt_3,306). happensAt(walking(mws1gt_3),306). orientation(mws1gt_5,168,306). coords(mws1gt_5,36,181,306).  visible(mws1gt_5,306). happensAt(walking(mws1gt_5),306).
orientation(mws1gt_3,158,307). coords(mws1gt_3,71,136,307).  visible(mws1gt_3,307). happensAt(walking(mws1gt_3),307). orientation(mws1gt_5,168,307). coords(mws1gt_5,36,182,307).  visible(mws1gt_5,307). happensAt(walking(mws1gt_5),307).
orientation(mws1gt_3,158,308). coords(mws1gt_3,70,137,308).  visible(mws1gt_3,308). happensAt(walking(mws1gt_3),308). orientation(mws1gt_5,168,308). coords(mws1gt_5,35,183,308).  visible(mws1gt_5,308). happensAt(walking(mws1gt_5),308).
orientation(mws1gt_3,158,309). coords(mws1gt_3,70,137,309).  visible(mws1gt_3,309). happensAt(walking(mws1gt_3),309). orientation(mws1gt_5,168,309). coords(mws1gt_5,35,185,309).  visible(mws1gt_5,309). happensAt(walking(mws1gt_5),309).
orientation(mws1gt_3,158,310). coords(mws1gt_3,69,137,310).  visible(mws1gt_3,310). happensAt(walking(mws1gt_3),310). orientation(mws1gt_5,168,310). coords(mws1gt_5,35,186,310).  visible(mws1gt_5,310). happensAt(walking(mws1gt_5),310).
#show holdsAt/2.