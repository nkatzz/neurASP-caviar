
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

	orientation(rwgt_0,42,63). coords(rwgt_0,281,261,63).  visible(rwgt_0,63). happensAt(walking(rwgt_0),63). orientation(rwgt_1,170,63). coords(rwgt_1,225,229,63).  visible(rwgt_1,63). happensAt(walking(rwgt_1),63).
orientation(rwgt_0,42,64). coords(rwgt_0,281,257,64).  visible(rwgt_0,64). happensAt(walking(rwgt_0),64). orientation(rwgt_1,170,64). coords(rwgt_1,222,225,64).  visible(rwgt_1,64). happensAt(walking(rwgt_1),64).
orientation(rwgt_0,55,65). coords(rwgt_0,280,253,65).  visible(rwgt_0,65). happensAt(walking(rwgt_0),65). orientation(rwgt_1,170,65). coords(rwgt_1,221,223,65).  visible(rwgt_1,65). happensAt(walking(rwgt_1),65).
orientation(rwgt_0,39,66). coords(rwgt_0,279,251,66).  visible(rwgt_0,66). happensAt(walking(rwgt_0),66). orientation(rwgt_1,170,66). coords(rwgt_1,218,220,66).  visible(rwgt_1,66). happensAt(walking(rwgt_1),66).
orientation(rwgt_0,44,67). coords(rwgt_0,279,250,67).  visible(rwgt_0,67). happensAt(walking(rwgt_0),67). orientation(rwgt_1,170,67). coords(rwgt_1,216,218,67).  visible(rwgt_1,67). happensAt(walking(rwgt_1),67).
orientation(rwgt_0,35,68). coords(rwgt_0,278,249,68).  visible(rwgt_0,68). happensAt(walking(rwgt_0),68). orientation(rwgt_1,170,68). coords(rwgt_1,216,216,68).  visible(rwgt_1,68). happensAt(walking(rwgt_1),68).
orientation(rwgt_0,28,69). coords(rwgt_0,278,249,69).  visible(rwgt_0,69). happensAt(walking(rwgt_0),69). orientation(rwgt_1,170,69). coords(rwgt_1,215,214,69).  visible(rwgt_1,69). happensAt(walking(rwgt_1),69).
orientation(rwgt_0,19,70). coords(rwgt_0,277,248,70).  visible(rwgt_0,70). happensAt(walking(rwgt_0),70). orientation(rwgt_1,170,70). coords(rwgt_1,215,211,70).  visible(rwgt_1,70). happensAt(walking(rwgt_1),70).
orientation(rwgt_0,2,71). coords(rwgt_0,276,247,71).  visible(rwgt_0,71). happensAt(walking(rwgt_0),71). orientation(rwgt_1,170,71). coords(rwgt_1,214,209,71).  visible(rwgt_1,71). happensAt(walking(rwgt_1),71).
orientation(rwgt_0,2,72). coords(rwgt_0,274,245,72).  visible(rwgt_0,72). happensAt(walking(rwgt_0),72). orientation(rwgt_1,167,72). coords(rwgt_1,213,209,72).  visible(rwgt_1,72). happensAt(walking(rwgt_1),72).
orientation(rwgt_0,2,73). coords(rwgt_0,273,243,73).  visible(rwgt_0,73). happensAt(walking(rwgt_0),73). orientation(rwgt_1,160,73). coords(rwgt_1,212,209,73).  visible(rwgt_1,73). happensAt(walking(rwgt_1),73).
orientation(rwgt_0,2,74). coords(rwgt_0,271,241,74).  visible(rwgt_0,74). happensAt(walking(rwgt_0),74). orientation(rwgt_1,159,74). coords(rwgt_1,210,206,74).  visible(rwgt_1,74). happensAt(walking(rwgt_1),74).
orientation(rwgt_0,2,75). coords(rwgt_0,270,239,75).  visible(rwgt_0,75). happensAt(walking(rwgt_0),75). orientation(rwgt_1,154,75). coords(rwgt_1,208,203,75).  visible(rwgt_1,75). happensAt(walking(rwgt_1),75).
orientation(rwgt_0,2,76). coords(rwgt_0,270,239,76).  visible(rwgt_0,76). happensAt(walking(rwgt_0),76). orientation(rwgt_1,157,76). coords(rwgt_1,206,200,76).  visible(rwgt_1,76). happensAt(walking(rwgt_1),76).
orientation(rwgt_0,2,77). coords(rwgt_0,269,237,77).  visible(rwgt_0,77). happensAt(walking(rwgt_0),77). orientation(rwgt_1,149,77). coords(rwgt_1,203,196,77).  visible(rwgt_1,77). happensAt(walking(rwgt_1),77).
orientation(rwgt_0,166,78). coords(rwgt_0,268,236,78).  visible(rwgt_0,78). happensAt(walking(rwgt_0),78). orientation(rwgt_1,149,78). coords(rwgt_1,201,193,78).  visible(rwgt_1,78). happensAt(walking(rwgt_1),78).
orientation(rwgt_0,166,79). coords(rwgt_0,267,234,79).  visible(rwgt_0,79). happensAt(walking(rwgt_0),79). orientation(rwgt_1,149,79). coords(rwgt_1,201,189,79).  visible(rwgt_1,79). happensAt(walking(rwgt_1),79).
orientation(rwgt_0,160,80). coords(rwgt_0,266,232,80).  visible(rwgt_0,80). happensAt(walking(rwgt_0),80). orientation(rwgt_1,149,80). coords(rwgt_1,200,188,80).  visible(rwgt_1,80). happensAt(walking(rwgt_1),80).
orientation(rwgt_0,160,81). coords(rwgt_0,265,230,81).  visible(rwgt_0,81). happensAt(walking(rwgt_0),81). orientation(rwgt_1,149,81). coords(rwgt_1,199,186,81).  visible(rwgt_1,81). happensAt(walking(rwgt_1),81).
orientation(rwgt_0,160,82). coords(rwgt_0,264,228,82).  visible(rwgt_0,82). happensAt(walking(rwgt_0),82). orientation(rwgt_1,149,82). coords(rwgt_1,198,185,82).  visible(rwgt_1,82). happensAt(walking(rwgt_1),82).
orientation(rwgt_0,160,83). coords(rwgt_0,263,226,83).  visible(rwgt_0,83). happensAt(walking(rwgt_0),83). orientation(rwgt_1,149,83). coords(rwgt_1,198,183,83).  visible(rwgt_1,83). happensAt(walking(rwgt_1),83).
orientation(rwgt_0,160,84). coords(rwgt_0,261,225,84).  visible(rwgt_0,84). happensAt(walking(rwgt_0),84). orientation(rwgt_1,149,84). coords(rwgt_1,197,182,84).  visible(rwgt_1,84). happensAt(walking(rwgt_1),84).
orientation(rwgt_0,160,85). coords(rwgt_0,260,223,85).  visible(rwgt_0,85). happensAt(walking(rwgt_0),85). orientation(rwgt_1,149,85). coords(rwgt_1,196,181,85).  visible(rwgt_1,85). happensAt(walking(rwgt_1),85).
orientation(rwgt_0,160,86). coords(rwgt_0,259,218,86).  visible(rwgt_0,86). happensAt(walking(rwgt_0),86). orientation(rwgt_1,149,86). coords(rwgt_1,195,179,86).  visible(rwgt_1,86). happensAt(walking(rwgt_1),86).
orientation(rwgt_0,160,87). coords(rwgt_0,255,216,87).  visible(rwgt_0,87). happensAt(walking(rwgt_0),87). orientation(rwgt_1,149,87). coords(rwgt_1,194,177,87).  visible(rwgt_1,87). happensAt(walking(rwgt_1),87).
#show holdsAt/2.