
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

	orientation(ricgt_3,51,51). coords(ricgt_3,301,286,51).  visible(ricgt_3,51). happensAt(inactive(ricgt_3),51). orientation(ricgt_5,0,51). coords(ricgt_5,30,119,51).  visible(ricgt_5,51). happensAt(walking(ricgt_5),51).
orientation(ricgt_3,80,52). coords(ricgt_3,292,285,52).  visible(ricgt_3,52). happensAt(walking(ricgt_3),52). orientation(ricgt_5,0,52). coords(ricgt_5,30,119,52).  visible(ricgt_5,52). happensAt(walking(ricgt_5),52).
orientation(ricgt_3,80,53). coords(ricgt_3,290,283,53).  visible(ricgt_3,53). happensAt(walking(ricgt_3),53). orientation(ricgt_5,0,53). coords(ricgt_5,30,119,53).  visible(ricgt_5,53). happensAt(walking(ricgt_5),53).
orientation(ricgt_3,77,54). coords(ricgt_3,281,280,54).  visible(ricgt_3,54). happensAt(walking(ricgt_3),54). orientation(ricgt_5,0,54). coords(ricgt_5,29,120,54).  visible(ricgt_5,54). happensAt(walking(ricgt_5),54).
orientation(ricgt_3,70,55). coords(ricgt_3,280,279,55).  visible(ricgt_3,55). happensAt(walking(ricgt_3),55). orientation(ricgt_5,0,55). coords(ricgt_5,29,120,55).  visible(ricgt_5,55). happensAt(walking(ricgt_5),55).
orientation(ricgt_3,62,56). coords(ricgt_3,279,277,56).  visible(ricgt_3,56). happensAt(walking(ricgt_3),56). orientation(ricgt_5,0,56). coords(ricgt_5,29,121,56).  visible(ricgt_5,56). happensAt(walking(ricgt_5),56).
orientation(ricgt_3,58,57). coords(ricgt_3,278,277,57).  visible(ricgt_3,57). happensAt(walking(ricgt_3),57). orientation(ricgt_5,0,57). coords(ricgt_5,29,121,57).  visible(ricgt_5,57). happensAt(walking(ricgt_5),57).
orientation(ricgt_3,50,58). coords(ricgt_3,278,277,58).  visible(ricgt_3,58). happensAt(walking(ricgt_3),58). orientation(ricgt_5,0,58). coords(ricgt_5,29,121,58).  visible(ricgt_5,58). happensAt(walking(ricgt_5),58).
orientation(ricgt_3,50,59). coords(ricgt_3,277,276,59).  visible(ricgt_3,59). happensAt(walking(ricgt_3),59). orientation(ricgt_5,0,59). coords(ricgt_5,28,122,59).  visible(ricgt_5,59). happensAt(walking(ricgt_5),59).
orientation(ricgt_3,48,60). coords(ricgt_3,276,276,60).  visible(ricgt_3,60). happensAt(walking(ricgt_3),60). orientation(ricgt_5,0,60). coords(ricgt_5,28,123,60).  visible(ricgt_5,60). happensAt(walking(ricgt_5),60).
orientation(ricgt_3,45,61). coords(ricgt_3,272,276,61).  visible(ricgt_3,61). happensAt(walking(ricgt_3),61). orientation(ricgt_5,0,61). coords(ricgt_5,28,123,61).  visible(ricgt_5,61). happensAt(walking(ricgt_5),61).
orientation(ricgt_3,44,62). coords(ricgt_3,270,276,62).  visible(ricgt_3,62). happensAt(walking(ricgt_3),62). orientation(ricgt_5,0,62). coords(ricgt_5,28,123,62).  visible(ricgt_5,62). happensAt(walking(ricgt_5),62).
orientation(ricgt_3,40,63). coords(ricgt_3,265,276,63).  visible(ricgt_3,63). happensAt(walking(ricgt_3),63). orientation(ricgt_5,0,63). coords(ricgt_5,28,123,63).  visible(ricgt_5,63). happensAt(walking(ricgt_5),63).
orientation(ricgt_3,36,64). coords(ricgt_3,262,274,64).  visible(ricgt_3,64). happensAt(walking(ricgt_3),64). orientation(ricgt_5,0,64). coords(ricgt_5,28,123,64).  visible(ricgt_5,64). happensAt(walking(ricgt_5),64).
orientation(ricgt_3,32,65). coords(ricgt_3,258,272,65).  visible(ricgt_3,65). happensAt(walking(ricgt_3),65). orientation(ricgt_5,0,65). coords(ricgt_5,28,123,65).  visible(ricgt_5,65). happensAt(walking(ricgt_5),65).
orientation(ricgt_3,26,66). coords(ricgt_3,257,267,66).  visible(ricgt_3,66). happensAt(walking(ricgt_3),66). orientation(ricgt_5,0,66). coords(ricgt_5,28,123,66).  visible(ricgt_5,66). happensAt(walking(ricgt_5),66).
orientation(ricgt_3,23,67). coords(ricgt_3,257,263,67).  visible(ricgt_3,67). happensAt(walking(ricgt_3),67). orientation(ricgt_5,0,67). coords(ricgt_5,28,123,67).  visible(ricgt_5,67). happensAt(walking(ricgt_5),67).
orientation(ricgt_3,34,68). coords(ricgt_3,256,259,68).  visible(ricgt_3,68). happensAt(walking(ricgt_3),68). orientation(ricgt_5,0,68). coords(ricgt_5,28,123,68).  visible(ricgt_5,68). happensAt(walking(ricgt_5),68).
orientation(ricgt_3,27,69). coords(ricgt_3,256,255,69).  visible(ricgt_3,69). happensAt(walking(ricgt_3),69). orientation(ricgt_5,0,69). coords(ricgt_5,28,123,69).  visible(ricgt_5,69). happensAt(walking(ricgt_5),69).
orientation(ricgt_3,27,70). coords(ricgt_3,255,255,70).  visible(ricgt_3,70). happensAt(walking(ricgt_3),70). orientation(ricgt_5,0,70). coords(ricgt_5,28,123,70).  visible(ricgt_5,70). happensAt(walking(ricgt_5),70).
orientation(ricgt_3,27,71). coords(ricgt_3,254,254,71).  visible(ricgt_3,71). happensAt(walking(ricgt_3),71). orientation(ricgt_5,0,71). coords(ricgt_5,28,123,71).  visible(ricgt_5,71). happensAt(walking(ricgt_5),71).
orientation(ricgt_3,21,72). coords(ricgt_3,251,253,72).  visible(ricgt_3,72). happensAt(walking(ricgt_3),72). orientation(ricgt_5,0,72). coords(ricgt_5,28,123,72).  visible(ricgt_5,72). happensAt(walking(ricgt_5),72).
orientation(ricgt_3,17,73). coords(ricgt_3,249,251,73).  visible(ricgt_3,73). happensAt(walking(ricgt_3),73). orientation(ricgt_5,0,73). coords(ricgt_5,28,124,73).  visible(ricgt_5,73). happensAt(walking(ricgt_5),73).
orientation(ricgt_3,15,74). coords(ricgt_3,247,246,74).  visible(ricgt_3,74). happensAt(walking(ricgt_3),74). orientation(ricgt_5,0,74). coords(ricgt_5,28,124,74).  visible(ricgt_5,74). happensAt(walking(ricgt_5),74).
orientation(ricgt_3,10,75). coords(ricgt_3,246,242,75).  visible(ricgt_3,75). happensAt(walking(ricgt_3),75). orientation(ricgt_5,0,75). coords(ricgt_5,28,124,75).  visible(ricgt_5,75). happensAt(walking(ricgt_5),75).
#show holdsAt/2.