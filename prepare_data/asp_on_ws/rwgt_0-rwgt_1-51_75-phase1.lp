
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

	orientation(rwgt_0,49,51). coords(rwgt_0,303,276,51).  visible(rwgt_0,51). happensAt(walking(rwgt_0),51). orientation(rwgt_1,3,51). coords(rwgt_1,241,257,51).  visible(rwgt_1,51). happensAt(walking(rwgt_1),51).
orientation(rwgt_0,51,52). coords(rwgt_0,302,275,52).  visible(rwgt_0,52). happensAt(walking(rwgt_0),52). orientation(rwgt_1,3,52). coords(rwgt_1,240,256,52).  visible(rwgt_1,52). happensAt(walking(rwgt_1),52).
orientation(rwgt_0,63,53). coords(rwgt_0,301,274,53).  visible(rwgt_0,53). happensAt(walking(rwgt_0),53). orientation(rwgt_1,164,53). coords(rwgt_1,239,254,53).  visible(rwgt_1,53). happensAt(walking(rwgt_1),53).
orientation(rwgt_0,61,54). coords(rwgt_0,300,274,54).  visible(rwgt_0,54). happensAt(walking(rwgt_0),54). orientation(rwgt_1,164,54). coords(rwgt_1,238,252,54).  visible(rwgt_1,54). happensAt(walking(rwgt_1),54).
orientation(rwgt_0,40,55). coords(rwgt_0,299,274,55).  visible(rwgt_0,55). happensAt(walking(rwgt_0),55). orientation(rwgt_1,164,55). coords(rwgt_1,237,249,55).  visible(rwgt_1,55). happensAt(walking(rwgt_1),55).
orientation(rwgt_0,42,56). coords(rwgt_0,298,274,56).  visible(rwgt_0,56). happensAt(walking(rwgt_0),56). orientation(rwgt_1,164,56). coords(rwgt_1,236,245,56).  visible(rwgt_1,56). happensAt(walking(rwgt_1),56).
orientation(rwgt_0,42,57). coords(rwgt_0,296,274,57).  visible(rwgt_0,57). happensAt(walking(rwgt_0),57). orientation(rwgt_1,170,57). coords(rwgt_1,235,243,57).  visible(rwgt_1,57). happensAt(walking(rwgt_1),57).
orientation(rwgt_0,42,58). coords(rwgt_0,294,274,58).  visible(rwgt_0,58). happensAt(walking(rwgt_0),58). orientation(rwgt_1,170,58). coords(rwgt_1,235,240,58).  visible(rwgt_1,58). happensAt(walking(rwgt_1),58).
orientation(rwgt_0,42,59). coords(rwgt_0,292,274,59).  visible(rwgt_0,59). happensAt(walking(rwgt_0),59). orientation(rwgt_1,170,59). coords(rwgt_1,234,238,59).  visible(rwgt_1,59). happensAt(walking(rwgt_1),59).
orientation(rwgt_0,42,60). coords(rwgt_0,289,271,60).  visible(rwgt_0,60). happensAt(walking(rwgt_0),60). orientation(rwgt_1,170,60). coords(rwgt_1,231,236,60).  visible(rwgt_1,60). happensAt(walking(rwgt_1),60).
orientation(rwgt_0,42,61). coords(rwgt_0,287,269,61).  visible(rwgt_0,61). happensAt(walking(rwgt_0),61). orientation(rwgt_1,170,61). coords(rwgt_1,230,235,61).  visible(rwgt_1,61). happensAt(walking(rwgt_1),61).
orientation(rwgt_0,42,62). coords(rwgt_0,284,266,62).  visible(rwgt_0,62). happensAt(walking(rwgt_0),62). orientation(rwgt_1,170,62). coords(rwgt_1,228,232,62).  visible(rwgt_1,62). happensAt(walking(rwgt_1),62).
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
#show holdsAt/2.