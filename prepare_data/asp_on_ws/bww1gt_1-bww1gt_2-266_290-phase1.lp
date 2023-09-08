
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

	orientation(bww1gt_1,168,266). coords(bww1gt_1,89,248,266).  visible(bww1gt_1,266). happensAt(walking(bww1gt_1),266). orientation(bww1gt_2,167,266). coords(bww1gt_2,47,165,266).  visible(bww1gt_2,266). happensAt(walking(bww1gt_2),266).
orientation(bww1gt_1,164,267). coords(bww1gt_1,89,248,267).  visible(bww1gt_1,267). happensAt(walking(bww1gt_1),267). orientation(bww1gt_2,167,267). coords(bww1gt_2,47,165,267).  visible(bww1gt_2,267). happensAt(walking(bww1gt_2),267).
orientation(bww1gt_1,164,268). coords(bww1gt_1,90,248,268).  visible(bww1gt_1,268). happensAt(walking(bww1gt_1),268). orientation(bww1gt_2,167,268). coords(bww1gt_2,47,165,268).  visible(bww1gt_2,268). happensAt(walking(bww1gt_2),268).
orientation(bww1gt_1,159,269). coords(bww1gt_1,90,249,269).  visible(bww1gt_1,269). happensAt(walking(bww1gt_1),269). orientation(bww1gt_2,167,269). coords(bww1gt_2,47,165,269).  visible(bww1gt_2,269). happensAt(walking(bww1gt_2),269).
orientation(bww1gt_1,157,270). coords(bww1gt_1,92,249,270).  visible(bww1gt_1,270). happensAt(walking(bww1gt_1),270). orientation(bww1gt_2,167,270). coords(bww1gt_2,47,165,270).  visible(bww1gt_2,270). happensAt(walking(bww1gt_2),270).
orientation(bww1gt_1,157,271). coords(bww1gt_1,92,249,271).  visible(bww1gt_1,271). happensAt(walking(bww1gt_1),271). orientation(bww1gt_2,167,271). coords(bww1gt_2,47,165,271).  visible(bww1gt_2,271). happensAt(walking(bww1gt_2),271).
orientation(bww1gt_1,156,272). coords(bww1gt_1,93,250,272).  visible(bww1gt_1,272). happensAt(walking(bww1gt_1),272). orientation(bww1gt_2,167,272). coords(bww1gt_2,47,165,272).  visible(bww1gt_2,272). happensAt(walking(bww1gt_2),272).
orientation(bww1gt_1,160,273). coords(bww1gt_1,93,251,273).  visible(bww1gt_1,273). happensAt(walking(bww1gt_1),273). orientation(bww1gt_2,167,273). coords(bww1gt_2,48,165,273).  visible(bww1gt_2,273). happensAt(walking(bww1gt_2),273).
orientation(bww1gt_1,158,274). coords(bww1gt_1,93,251,274).  visible(bww1gt_1,274). happensAt(walking(bww1gt_1),274). orientation(bww1gt_2,167,274). coords(bww1gt_2,49,165,274).  visible(bww1gt_2,274). happensAt(walking(bww1gt_2),274).
orientation(bww1gt_1,168,275). coords(bww1gt_1,94,252,275).  visible(bww1gt_1,275). happensAt(walking(bww1gt_1),275). orientation(bww1gt_2,167,275). coords(bww1gt_2,49,165,275).  visible(bww1gt_2,275). happensAt(walking(bww1gt_2),275).
orientation(bww1gt_1,167,276). coords(bww1gt_1,94,253,276).  visible(bww1gt_1,276). happensAt(walking(bww1gt_1),276). orientation(bww1gt_2,167,276). coords(bww1gt_2,49,165,276).  visible(bww1gt_2,276). happensAt(walking(bww1gt_2),276).
orientation(bww1gt_1,165,277). coords(bww1gt_1,94,254,277).  visible(bww1gt_1,277). happensAt(walking(bww1gt_1),277). orientation(bww1gt_2,167,277). coords(bww1gt_2,50,165,277).  visible(bww1gt_2,277). happensAt(walking(bww1gt_2),277).
orientation(bww1gt_1,166,278). coords(bww1gt_1,95,256,278).  visible(bww1gt_1,278). happensAt(walking(bww1gt_1),278). orientation(bww1gt_2,167,278). coords(bww1gt_2,50,165,278).  visible(bww1gt_2,278). happensAt(walking(bww1gt_2),278).
orientation(bww1gt_1,173,279). coords(bww1gt_1,95,256,279).  visible(bww1gt_1,279). happensAt(walking(bww1gt_1),279). orientation(bww1gt_2,167,279). coords(bww1gt_2,51,165,279).  visible(bww1gt_2,279). happensAt(walking(bww1gt_2),279).
orientation(bww1gt_1,173,280). coords(bww1gt_1,95,257,280).  visible(bww1gt_1,280). happensAt(walking(bww1gt_1),280). orientation(bww1gt_2,167,280). coords(bww1gt_2,51,165,280).  visible(bww1gt_2,280). happensAt(walking(bww1gt_2),280).
orientation(bww1gt_1,173,281). coords(bww1gt_1,95,257,281).  visible(bww1gt_1,281). happensAt(walking(bww1gt_1),281). orientation(bww1gt_2,167,281). coords(bww1gt_2,51,165,281).  visible(bww1gt_2,281). happensAt(walking(bww1gt_2),281).
orientation(bww1gt_1,173,282). coords(bww1gt_1,95,257,282).  visible(bww1gt_1,282). happensAt(walking(bww1gt_1),282). orientation(bww1gt_2,167,282). coords(bww1gt_2,51,165,282).  visible(bww1gt_2,282). happensAt(walking(bww1gt_2),282).
orientation(bww1gt_1,173,283). coords(bww1gt_1,95,258,283).  visible(bww1gt_1,283). happensAt(walking(bww1gt_1),283). orientation(bww1gt_2,167,283). coords(bww1gt_2,51,165,283).  visible(bww1gt_2,283). happensAt(walking(bww1gt_2),283).
orientation(bww1gt_1,173,284). coords(bww1gt_1,95,258,284).  visible(bww1gt_1,284). happensAt(walking(bww1gt_1),284). orientation(bww1gt_2,167,284). coords(bww1gt_2,51,165,284).  visible(bww1gt_2,284). happensAt(walking(bww1gt_2),284).
orientation(bww1gt_1,173,285). coords(bww1gt_1,95,259,285).  visible(bww1gt_1,285). happensAt(walking(bww1gt_1),285). orientation(bww1gt_2,167,285). coords(bww1gt_2,52,165,285).  visible(bww1gt_2,285). happensAt(walking(bww1gt_2),285).
orientation(bww1gt_1,173,286). coords(bww1gt_1,97,260,286).  visible(bww1gt_1,286). happensAt(walking(bww1gt_1),286). orientation(bww1gt_2,167,286). coords(bww1gt_2,52,165,286).  visible(bww1gt_2,286). happensAt(walking(bww1gt_2),286).
orientation(bww1gt_1,173,287). coords(bww1gt_1,99,260,287).  visible(bww1gt_1,287). happensAt(walking(bww1gt_1),287). orientation(bww1gt_2,167,287). coords(bww1gt_2,52,165,287).  visible(bww1gt_2,287). happensAt(walking(bww1gt_2),287).
orientation(bww1gt_1,172,288). coords(bww1gt_1,100,261,288).  visible(bww1gt_1,288). happensAt(walking(bww1gt_1),288). orientation(bww1gt_2,167,288). coords(bww1gt_2,53,164,288).  visible(bww1gt_2,288). happensAt(walking(bww1gt_2),288).
orientation(bww1gt_1,0,289). coords(bww1gt_1,101,261,289).  visible(bww1gt_1,289). happensAt(inactive(bww1gt_1),289). orientation(bww1gt_2,167,289). coords(bww1gt_2,53,164,289).  visible(bww1gt_2,289). happensAt(walking(bww1gt_2),289).
orientation(bww1gt_1,7,290). coords(bww1gt_1,101,262,290).  visible(bww1gt_1,290). happensAt(inactive(bww1gt_1),290). orientation(bww1gt_2,167,290). coords(bww1gt_2,54,164,290).  visible(bww1gt_2,290). happensAt(walking(bww1gt_2),290).
#show holdsAt/2.