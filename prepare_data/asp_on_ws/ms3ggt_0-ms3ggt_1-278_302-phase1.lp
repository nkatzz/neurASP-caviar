
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

	orientation(ms3ggt_0,161,278). coords(ms3ggt_0,68,149,278).  visible(ms3ggt_0,278). happensAt(walking(ms3ggt_0),278). orientation(ms3ggt_1,158,278). coords(ms3ggt_1,66,164,278).  visible(ms3ggt_1,278). happensAt(walking(ms3ggt_1),278).
orientation(ms3ggt_0,161,279). coords(ms3ggt_0,69,149,279).  visible(ms3ggt_0,279). happensAt(walking(ms3ggt_0),279). orientation(ms3ggt_1,158,279). coords(ms3ggt_1,68,164,279).  visible(ms3ggt_1,279). happensAt(walking(ms3ggt_1),279).
orientation(ms3ggt_0,161,280). coords(ms3ggt_0,69,149,280).  visible(ms3ggt_0,280). happensAt(walking(ms3ggt_0),280). orientation(ms3ggt_1,158,280). coords(ms3ggt_1,68,164,280).  visible(ms3ggt_1,280). happensAt(walking(ms3ggt_1),280).
orientation(ms3ggt_0,161,281). coords(ms3ggt_0,69,149,281).  visible(ms3ggt_0,281). happensAt(walking(ms3ggt_0),281). orientation(ms3ggt_1,158,281). coords(ms3ggt_1,68,164,281).  visible(ms3ggt_1,281). happensAt(walking(ms3ggt_1),281).
orientation(ms3ggt_0,153,282). coords(ms3ggt_0,69,148,282).  visible(ms3ggt_0,282). happensAt(walking(ms3ggt_0),282). orientation(ms3ggt_1,158,282). coords(ms3ggt_1,69,164,282).  visible(ms3ggt_1,282). happensAt(walking(ms3ggt_1),282).
orientation(ms3ggt_0,153,283). coords(ms3ggt_0,69,148,283).  visible(ms3ggt_0,283). happensAt(walking(ms3ggt_0),283). orientation(ms3ggt_1,158,283). coords(ms3ggt_1,69,164,283).  visible(ms3ggt_1,283). happensAt(walking(ms3ggt_1),283).
orientation(ms3ggt_0,153,284). coords(ms3ggt_0,69,148,284).  visible(ms3ggt_0,284). happensAt(walking(ms3ggt_0),284). orientation(ms3ggt_1,158,284). coords(ms3ggt_1,69,164,284).  visible(ms3ggt_1,284). happensAt(walking(ms3ggt_1),284).
orientation(ms3ggt_0,153,285). coords(ms3ggt_0,69,148,285).  visible(ms3ggt_0,285). happensAt(walking(ms3ggt_0),285). orientation(ms3ggt_1,158,285). coords(ms3ggt_1,70,164,285).  visible(ms3ggt_1,285). happensAt(walking(ms3ggt_1),285).
orientation(ms3ggt_0,153,286). coords(ms3ggt_0,69,146,286).  visible(ms3ggt_0,286). happensAt(walking(ms3ggt_0),286). orientation(ms3ggt_1,158,286). coords(ms3ggt_1,70,164,286).  visible(ms3ggt_1,286). happensAt(walking(ms3ggt_1),286).
orientation(ms3ggt_0,153,287). coords(ms3ggt_0,73,146,287).  visible(ms3ggt_0,287). happensAt(walking(ms3ggt_0),287). orientation(ms3ggt_1,158,287). coords(ms3ggt_1,70,164,287).  visible(ms3ggt_1,287). happensAt(walking(ms3ggt_1),287).
orientation(ms3ggt_0,153,288). coords(ms3ggt_0,74,146,288).  visible(ms3ggt_0,288). happensAt(walking(ms3ggt_0),288). orientation(ms3ggt_1,158,288). coords(ms3ggt_1,70,164,288).  visible(ms3ggt_1,288). happensAt(walking(ms3ggt_1),288).
orientation(ms3ggt_0,153,289). coords(ms3ggt_0,74,146,289).  visible(ms3ggt_0,289). happensAt(walking(ms3ggt_0),289). orientation(ms3ggt_1,158,289). coords(ms3ggt_1,71,164,289).  visible(ms3ggt_1,289). happensAt(walking(ms3ggt_1),289).
orientation(ms3ggt_0,159,290). coords(ms3ggt_0,76,146,290).  visible(ms3ggt_0,290). happensAt(walking(ms3ggt_0),290). orientation(ms3ggt_1,158,290). coords(ms3ggt_1,72,164,290).  visible(ms3ggt_1,290). happensAt(walking(ms3ggt_1),290).
orientation(ms3ggt_0,159,291). coords(ms3ggt_0,76,146,291).  visible(ms3ggt_0,291). happensAt(walking(ms3ggt_0),291). orientation(ms3ggt_1,158,291). coords(ms3ggt_1,72,164,291).  visible(ms3ggt_1,291). happensAt(walking(ms3ggt_1),291).
orientation(ms3ggt_0,159,292). coords(ms3ggt_0,76,146,292).  visible(ms3ggt_0,292). happensAt(walking(ms3ggt_0),292). orientation(ms3ggt_1,158,292). coords(ms3ggt_1,73,164,292).  visible(ms3ggt_1,292). happensAt(walking(ms3ggt_1),292).
orientation(ms3ggt_0,159,293). coords(ms3ggt_0,77,146,293).  visible(ms3ggt_0,293). happensAt(walking(ms3ggt_0),293). orientation(ms3ggt_1,158,293). coords(ms3ggt_1,75,164,293).  visible(ms3ggt_1,293). happensAt(walking(ms3ggt_1),293).
orientation(ms3ggt_0,159,294). coords(ms3ggt_0,77,146,294).  visible(ms3ggt_0,294). happensAt(walking(ms3ggt_0),294). orientation(ms3ggt_1,158,294). coords(ms3ggt_1,76,164,294).  visible(ms3ggt_1,294). happensAt(walking(ms3ggt_1),294).
orientation(ms3ggt_0,159,295). coords(ms3ggt_0,77,146,295).  visible(ms3ggt_0,295). happensAt(walking(ms3ggt_0),295). orientation(ms3ggt_1,158,295). coords(ms3ggt_1,76,164,295).  visible(ms3ggt_1,295). happensAt(walking(ms3ggt_1),295).
orientation(ms3ggt_0,159,296). coords(ms3ggt_0,77,145,296).  visible(ms3ggt_0,296). happensAt(walking(ms3ggt_0),296). orientation(ms3ggt_1,158,296). coords(ms3ggt_1,76,164,296).  visible(ms3ggt_1,296). happensAt(walking(ms3ggt_1),296).
orientation(ms3ggt_0,159,297). coords(ms3ggt_0,77,145,297).  visible(ms3ggt_0,297). happensAt(walking(ms3ggt_0),297). orientation(ms3ggt_1,158,297). coords(ms3ggt_1,76,164,297).  visible(ms3ggt_1,297). happensAt(walking(ms3ggt_1),297).
orientation(ms3ggt_0,159,298). coords(ms3ggt_0,77,145,298).  visible(ms3ggt_0,298). happensAt(walking(ms3ggt_0),298). orientation(ms3ggt_1,158,298). coords(ms3ggt_1,76,164,298).  visible(ms3ggt_1,298). happensAt(walking(ms3ggt_1),298).
orientation(ms3ggt_0,159,299). coords(ms3ggt_0,77,145,299).  visible(ms3ggt_0,299). happensAt(walking(ms3ggt_0),299). orientation(ms3ggt_1,158,299). coords(ms3ggt_1,77,164,299).  visible(ms3ggt_1,299). happensAt(walking(ms3ggt_1),299).
orientation(ms3ggt_0,159,300). coords(ms3ggt_0,78,145,300).  visible(ms3ggt_0,300). happensAt(walking(ms3ggt_0),300). orientation(ms3ggt_1,158,300). coords(ms3ggt_1,77,164,300).  visible(ms3ggt_1,300). happensAt(walking(ms3ggt_1),300).
orientation(ms3ggt_0,159,301). coords(ms3ggt_0,80,144,301).  visible(ms3ggt_0,301). happensAt(walking(ms3ggt_0),301). orientation(ms3ggt_1,158,301). coords(ms3ggt_1,77,164,301).  visible(ms3ggt_1,301). happensAt(walking(ms3ggt_1),301).
orientation(ms3ggt_0,159,302). coords(ms3ggt_0,81,144,302).  visible(ms3ggt_0,302). happensAt(walking(ms3ggt_0),302). orientation(ms3ggt_1,158,302). coords(ms3ggt_1,78,164,302).  visible(ms3ggt_1,302). happensAt(walking(ms3ggt_1),302).
#show holdsAt/2.