
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

	orientation(fomdgt3_1,150,276). coords(fomdgt3_1,53,96,276).  visible(fomdgt3_1,276). happensAt(walking(fomdgt3_1),276). orientation(fomdgt3_2,130,276). coords(fomdgt3_2,235,132,276).  visible(fomdgt3_2,276). happensAt(walking(fomdgt3_2),276).
orientation(fomdgt3_1,150,277). coords(fomdgt3_1,51,96,277).  visible(fomdgt3_1,277). happensAt(walking(fomdgt3_1),277). orientation(fomdgt3_2,130,277). coords(fomdgt3_2,235,130,277).  visible(fomdgt3_2,277). happensAt(walking(fomdgt3_2),277).
orientation(fomdgt3_1,150,278). coords(fomdgt3_1,51,96,278).  visible(fomdgt3_1,278). happensAt(walking(fomdgt3_1),278). orientation(fomdgt3_2,130,278). coords(fomdgt3_2,234,129,278).  visible(fomdgt3_2,278). happensAt(walking(fomdgt3_2),278).
orientation(fomdgt3_1,150,279). coords(fomdgt3_1,51,96,279).  visible(fomdgt3_1,279). happensAt(walking(fomdgt3_1),279). orientation(fomdgt3_2,130,279). coords(fomdgt3_2,234,129,279).  visible(fomdgt3_2,279). happensAt(walking(fomdgt3_2),279).
orientation(fomdgt3_1,150,280). coords(fomdgt3_1,51,96,280).  visible(fomdgt3_1,280). happensAt(walking(fomdgt3_1),280). orientation(fomdgt3_2,130,280). coords(fomdgt3_2,234,128,280).  visible(fomdgt3_2,280). happensAt(walking(fomdgt3_2),280).
orientation(fomdgt3_1,150,281). coords(fomdgt3_1,51,96,281).  visible(fomdgt3_1,281). happensAt(walking(fomdgt3_1),281). orientation(fomdgt3_2,130,281). coords(fomdgt3_2,232,127,281).  visible(fomdgt3_2,281). happensAt(walking(fomdgt3_2),281).
orientation(fomdgt3_1,150,282). coords(fomdgt3_1,51,96,282).  visible(fomdgt3_1,282). happensAt(walking(fomdgt3_1),282). orientation(fomdgt3_2,130,282). coords(fomdgt3_2,227,127,282).  visible(fomdgt3_2,282). happensAt(walking(fomdgt3_2),282).
orientation(fomdgt3_1,150,283). coords(fomdgt3_1,51,96,283).  visible(fomdgt3_1,283). happensAt(walking(fomdgt3_1),283). orientation(fomdgt3_2,130,283). coords(fomdgt3_2,227,127,283).  visible(fomdgt3_2,283). happensAt(walking(fomdgt3_2),283).
orientation(fomdgt3_1,150,284). coords(fomdgt3_1,51,96,284).  visible(fomdgt3_1,284). happensAt(walking(fomdgt3_1),284). orientation(fomdgt3_2,130,284). coords(fomdgt3_2,224,127,284).  visible(fomdgt3_2,284). happensAt(walking(fomdgt3_2),284).
orientation(fomdgt3_1,150,285). coords(fomdgt3_1,51,96,285).  visible(fomdgt3_1,285). happensAt(walking(fomdgt3_1),285). orientation(fomdgt3_2,130,285). coords(fomdgt3_2,224,127,285).  visible(fomdgt3_2,285). happensAt(walking(fomdgt3_2),285).
orientation(fomdgt3_1,150,286). coords(fomdgt3_1,51,96,286).  visible(fomdgt3_1,286). happensAt(active(fomdgt3_1),286). orientation(fomdgt3_2,130,286). coords(fomdgt3_2,224,125,286).  visible(fomdgt3_2,286). happensAt(walking(fomdgt3_2),286).
orientation(fomdgt3_1,150,287). coords(fomdgt3_1,51,96,287).  visible(fomdgt3_1,287). happensAt(active(fomdgt3_1),287). orientation(fomdgt3_2,130,287). coords(fomdgt3_2,224,123,287).  visible(fomdgt3_2,287). happensAt(walking(fomdgt3_2),287).
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
#show holdsAt/2.